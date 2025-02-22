from __future__ import absolute_import, division, print_function
import datetime
import os
import os.path as osp
import time
from collections import namedtuple, OrderedDict
from copy import deepcopy
from torchreid.utils.tools import StateCacher, set_random_seed
import optuna

import numpy as np
import torch
from torch.nn import functional as F
from torch.optim.lr_scheduler import OneCycleLR

from torchreid.integration.nncf.compression import (get_nncf_complession_stage,
                                                    get_nncf_prepare_for_tensorboard)
from torchreid.optim import ReduceLROnPlateauV2, WarmupScheduler, CosineAnnealingCycleRestart
from torchreid import metrics
from torchreid.losses import DeepSupervision
from torchreid.utils import (AverageMeter, MetricMeter, get_model_attr,
                             open_all_layers, open_specified_layers,
                             re_ranking, save_checkpoint,
                             visualize_ranked_results, ModelEmaV2)


EpochIntervalToValue = namedtuple('EpochIntervalToValue', ['first', 'last', 'value_inside', 'value_outside'])

def _get_cur_action_from_epoch_interval(epoch_interval, epoch):
    assert isinstance(epoch_interval, EpochIntervalToValue)
    if epoch_interval.first is None and epoch_interval.last is None:
        raise RuntimeError(f'Wrong epoch_interval {epoch_interval}')

    if epoch_interval.first is not None and epoch < epoch_interval.first:
        return epoch_interval.value_outside
    if epoch_interval.last is not None and epoch > epoch_interval.last:
        return epoch_interval.value_outside

    return epoch_interval.value_inside

def get_initial_lr_from_checkpoint(filename):
    if not filename:
        return None
    checkpoint = torch.load(filename, map_location='cpu')
    if not isinstance(checkpoint, dict):
        return None
    return checkpoint.get('initial_lr')

class Engine:
    r"""A generic base Engine class for both image- and video-reid."""
    def __init__(self,
                 datamanager,
                 models,
                 optimizers,
                 schedulers,
                 use_gpu=True,
                 save_all_chkpts=True,
                 train_patience = 10,
                 lr_decay_factor = 1000,
                 lr_finder = None,
                 early_stoping=False,
                 should_freeze_aux_models=False,
                 nncf_metainfo=None,
                 compression_ctrl=None,
                 initial_lr=None,
                 target_metric = 'train_loss',
                 epoch_interval_for_aux_model_freeze=None,
                 epoch_interval_for_turn_off_mutual_learning=None,
                 use_ema_decay=False,
                 ema_decay=0.999,
                 seed=5):

        self.datamanager = datamanager
        self.train_loader = self.datamanager.train_loader
        self.test_loader = self.datamanager.test_loader
        self.use_gpu = (torch.cuda.is_available() and use_gpu)
        self.save_all_chkpts = save_all_chkpts
        self.writer = None
        self.use_ema_decay = use_ema_decay
        self.start_epoch = 0
        self.lr_finder = lr_finder
        self.fixbase_epoch = 0
        self.iter_to_wait = 0
        self.best_metric = 0.0
        self.max_epoch = None
        self.num_batches = None
        assert target_metric in ['train_loss', 'test_acc']
        self.target_metric = target_metric
        self.epoch = None
        self.train_patience = train_patience
        self.early_stoping = early_stoping
        self.state_cacher = StateCacher(in_memory=True, cache_dir=None)
        self.param_history = set()
        self.seed = seed
        self.models = OrderedDict()
        self.optims = OrderedDict()
        self.scheds = OrderedDict()
        self.ema_model = None
        if should_freeze_aux_models:
            print(f'Engine: should_freeze_aux_models={should_freeze_aux_models}')
        self.should_freeze_aux_models = should_freeze_aux_models
        self.nncf_metainfo = deepcopy(nncf_metainfo)
        self.compression_ctrl = compression_ctrl
        self.initial_lr = initial_lr
        self.epoch_interval_for_aux_model_freeze = epoch_interval_for_aux_model_freeze
        self.epoch_interval_for_turn_off_mutual_learning = epoch_interval_for_turn_off_mutual_learning
        self.model_names_to_freeze = []
        self.current_lr = None

        if isinstance(models, (tuple, list)):
            assert isinstance(optimizers, (tuple, list))
            assert isinstance(schedulers, (tuple, list))

            num_models = len(models)
            assert len(optimizers) == num_models
            assert len(schedulers) == num_models

            for model_id, (model, optimizer, scheduler) in enumerate(zip(models, optimizers, schedulers)):
                model_name = 'main_model' if model_id == 0 else f'aux_model_{model_id}'
                self.register_model(model_name, model, optimizer, scheduler)
                if use_ema_decay and model_id == 0:
                    self.ema_model = ModelEmaV2(model, decay=ema_decay)
                if should_freeze_aux_models and model_id > 0:
                    self.model_names_to_freeze.append(model_name)
        else:
            assert not isinstance(optimizers, (tuple, list))
            assert not isinstance(schedulers, (tuple, list))
            assert not isinstance(models, (tuple, list))
            self.register_model('main_model', models, optimizers, schedulers)
            if use_ema_decay:
                self.ema_model = ModelEmaV2(models, decay=ema_decay)
        self.main_model_name = self.get_model_names()[0]
        assert initial_lr is not None
        self.lb_lr = initial_lr / lr_decay_factor
        self.per_batch_annealing = isinstance(self.scheds[self.main_model_name], (CosineAnnealingCycleRestart, OneCycleLR))

    def _should_freeze_aux_models(self, epoch):
        if not self.should_freeze_aux_models:
            return False
        if self.epoch_interval_for_aux_model_freeze is None:
            # simple case
            return True
        res = _get_cur_action_from_epoch_interval(self.epoch_interval_for_aux_model_freeze, epoch)
        print(f'_should_freeze_aux_models: return res={res}')
        return res

    def _should_turn_off_mutual_learning(self, epoch):
        if self.epoch_interval_for_turn_off_mutual_learning is None:
            # simple case
            return False
        res = _get_cur_action_from_epoch_interval(self.epoch_interval_for_turn_off_mutual_learning, epoch)
        print(f'_should_turn_off_mutual_learning: return {res}')
        return res

    def register_model(self, name='main_model', model=None, optim=None, sched=None):
        if self.__dict__.get('models') is None:
            raise AttributeError(
                'Cannot assign model before super().__init__() call'
            )

        if self.__dict__.get('optims') is None:
            raise AttributeError(
                'Cannot assign optim before super().__init__() call'
            )

        if self.__dict__.get('scheds') is None:
            raise AttributeError(
                'Cannot assign sched before super().__init__() call'
            )

        self.models[name] = model
        self.optims[name] = optim
        self.scheds[name] = sched

    def get_model_names(self, names=None):
        names_real = list(self.models.keys())
        if names is not None:
            if not isinstance(names, list):
                names = [names]
            for name in names:
                assert name in names_real
            return names
        else:
            return names_real

    def save_model(self, epoch, save_dir, is_best=False, should_save_ema_model=False):
        def create_sym_link(path,name):
            if osp.lexists(name):
                os.remove(name)
            os.symlink(path, name)

        names = self.get_model_names()
        for name in names:
            if should_save_ema_model and name == self.main_model_name:
                assert self.use_ema_decay
                model_state_dict = self.ema_model.module.state_dict()
            else:
                model_state_dict = self.models[name].state_dict()

            ckpt_path = save_checkpoint(
                            {
                                'state_dict': model_state_dict,
                                'epoch': epoch + 1,
                                'optimizer': self.optims[name].state_dict(),
                                'scheduler': self.scheds[name].state_dict(),
                                'num_classes': self.datamanager.num_train_pids,
                                'classes_map': self.datamanager.train_loader.dataset.classes,
                                'nncf_metainfo': self.nncf_metainfo,
                                'initial_lr': self.initial_lr
                            },
                            osp.join(save_dir, name),
                            is_best=is_best,
                            name=name
                        )

            if name == self.main_model_name:
                latest_name = osp.join(save_dir, 'latest.pth')
                create_sym_link(ckpt_path, latest_name)
                if is_best:
                    best_model = osp.join(save_dir, 'best.pth')
                    create_sym_link(ckpt_path, best_model)

    def set_model_mode(self, mode='train', names=None):
        assert mode in ['train', 'eval', 'test']
        names = self.get_model_names(names)

        for name in names:
            if mode == 'train':
                self.models[name].train()
            else:
                self.models[name].eval()

    def get_current_lr(self, names=None):
        names = self.get_model_names(names)
        name = names[0]
        return self.optims[name].param_groups[0]['lr']

    def update_lr(self, names=None, output_avg_metric=None):
        names = self.get_model_names(names)

        for name in names:
            if self.scheds[name] is not None:
                if isinstance(self.scheds[name], (ReduceLROnPlateauV2, WarmupScheduler)):
                    self.scheds[name].step(metrics=output_avg_metric)
                else:
                    self.scheds[name].step()

    def exit_on_plateau_and_choose_best(self, top1, smooth_top1):
        '''
        The function returns a pair (should_exit, is_candidate_for_best).

        Default implementation of the method returns False for should_exit.
        Other behavior must be overridden in derived classes from the base Engine.
        '''

        is_candidate_for_best = False
        current_metric = np.round(top1, 4)
        if current_metric >= self.best_metric:
            self.best_metric = current_metric
            is_candidate_for_best = True

        return False, is_candidate_for_best

    def run(
        self,
        trial=None,
        save_dir='log',
        tb_writer=None,
        max_epoch=0,
        start_epoch=0,
        print_freq=10,
        fixbase_epoch=0,
        open_layers=None,
        start_eval=0,
        eval_freq=-1,
        dist_metric='euclidean',
        normalize_feature=False,
        visrank=False,
        visrank_topk=10,
        use_metric_cuhk03=False,
        ranks=(1, 5, 10, 20),
        lr_finder=None,
        perf_monitor=None,
        stop_callback=None,
        initial_seed=5,
        **kwargs
    ):
        r"""A unified pipeline for training and evaluating a model.

        Args:
            save_dir (str): directory to save model.
            max_epoch (int): maximum epoch.
            start_epoch (int, optional): starting epoch. Default is 0.
            print_freq (int, optional): print_frequency. Default is 10.
            fixbase_epoch (int, optional): number of epochs to train ``open_layers`` (new layers)
                while keeping base layers frozen. Default is 0. ``fixbase_epoch`` is counted
                in ``max_epoch``.
            open_layers (str or list, optional): layers (attribute names) open for training.
            start_eval (int, optional): from which epoch to start evaluation. Default is 0.
            eval_freq (int, optional): evaluation frequency. Default is -1 (meaning evaluation
                is only performed at the end of training).
            dist_metric (str, optional): distance metric used to compute distance matrix
                between query and gallery. Default is "euclidean".
            normalize_feature (bool, optional): performs L2 normalization on feature vectors before
                computing feature distance. Default is False.
            visrank (bool, optional): visualizes ranked results. Default is False. It is recommended to
                enable ``visrank`` when ``test_only`` is True. The ranked images will be saved to
                "save_dir/visrank_dataset", e.g. "save_dir/visrank_market1501".
            visrank_topk (int, optional): top-k ranked images to be visualized. Default is 10.
            use_metric_cuhk03 (bool, optional): use single-gallery-shot setting for cuhk03.
                Default is False. This should be enabled when using cuhk03 classic split.
            ranks (list, optional): cmc ranks to be computed. Default is [1, 5, 10, 20].
            rerank (bool, optional): uses person re-ranking (by Zhong et al. CVPR'17).
                Default is False. This is only enabled when test_only=True.
        """

        if lr_finder:
            self.configure_lr_finder(trial, lr_finder)
            self.backup_model()

        self.writer = tb_writer
        time_start = time.time()
        self.start_epoch = start_epoch
        self.max_epoch = max_epoch
        assert start_epoch != max_epoch, "the last epoch number cannot be equal the start one"
        if self.early_stoping or self.target_metric == 'test_acc':
            assert eval_freq == 1, "early stopping works only with evaluation on each epoch" 
        self.fixbase_epoch = fixbase_epoch
        test_acc = AverageMeter()
        top1, smooth_top1, should_save_ema_model = 0, 0, False
        print('=> Start training')

        if perf_monitor and not lr_finder: perf_monitor.on_train_begin()
        for self.epoch in range(self.start_epoch, self.max_epoch):
            # change the NumPy’s seed at every epoch
            np.random.seed(initial_seed + self.epoch)
            if perf_monitor and not lr_finder: perf_monitor.on_train_epoch_begin()
            if self.compression_ctrl is not None:
                self.compression_ctrl.scheduler.epoch_step(self.epoch)
            avg_loss = self.train(
                print_freq=print_freq,
                fixbase_epoch=fixbase_epoch,
                open_layers=open_layers,
                lr_finder=lr_finder,
                perf_monitor=perf_monitor,
                stop_callback=stop_callback
            )
            if self.compression_ctrl is not None:
                statistics = self.compression_ctrl.statistics()
                print(statistics.to_str())
                if self.writer is not None and not lr_finder:
                    for key, value in get_nncf_prepare_for_tensorboard()(statistics).items():
                        self.writer.add_scalar("compression/statistics/{0}".format(key),
                                               value, len(self.train_loader) * self.epoch)

            if stop_callback and stop_callback.check_stop():
                break

            if perf_monitor and not lr_finder: perf_monitor.on_train_epoch_end()

            if (((self.epoch + 1) >= start_eval
               and eval_freq > 0
               and (self.epoch+1) % eval_freq == 0
               and (self.epoch + 1) != self.max_epoch)
               or self.epoch == (self.max_epoch - 1)):

                top1, should_save_ema_model = self.test(
                    self.epoch,
                    dist_metric=dist_metric,
                    normalize_feature=normalize_feature,
                    visrank=visrank,
                    visrank_topk=visrank_topk,
                    save_dir=save_dir,
                    use_metric_cuhk03=use_metric_cuhk03,
                    ranks=ranks,
                    lr_finder=lr_finder,
                )
            if top1:
                test_acc.update(top1) 
                smooth_top1 = test_acc.avg
            target_metric = smooth_top1 if self.target_metric == 'test_acc' else avg_loss

            if not lr_finder and not self.per_batch_annealing:
                self.update_lr(output_avg_metric = target_metric)

            if lr_finder:
                print(f"epoch: {self.epoch}\t top1: {top1}\t lr: {self.get_current_lr()}")
                if trial:
                    trial.report(top1, self.epoch)
                    if trial.should_prune():
                        # restore model before pruning
                        self.restore_model()
                        raise optuna.exceptions.TrialPruned()

            if not lr_finder:
                # use smooth (average) top1 metric for early stopping if the target metric is top1
                should_exit, is_candidate_for_best = self.exit_on_plateau_and_choose_best(top1, smooth_top1)
                should_exit = self.early_stoping and should_exit

                if self.save_all_chkpts:
                    self.save_model(self.epoch, save_dir, is_best=is_candidate_for_best,
                                    should_save_ema_model=should_save_ema_model)
                elif is_candidate_for_best:
                    self.save_model(0, save_dir, is_best=is_candidate_for_best,
                                    should_save_ema_model=should_save_ema_model)

                if should_exit:
                    if self.compression_ctrl is None or \
                            (self.compression_ctrl is not None and
                                self.compression_ctrl.compression_stage == get_nncf_complession_stage().FULLY_COMPRESSED):
                        break

        if perf_monitor and not lr_finder: perf_monitor.on_train_end()
        if lr_finder and lr_finder.mode != 'fast_ai': self.restore_model()
        elapsed = round(time.time() - time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print('Elapsed {}'.format(elapsed))

        if self.writer is not None:
            self.writer.close()

        return top1

    def _freeze_aux_models(self):
        for model_name in self.model_names_to_freeze:
            model = self.models[model_name]
            model.eval()
            open_specified_layers(model, [])

    def _unfreeze_aux_models(self):
        for model_name in self.model_names_to_freeze:
            model = self.models[model_name]
            model.train()
            open_all_layers(model)

    def configure_lr_finder(self, trial, finder_cfg):
        if trial is None:
            return
        lr = trial.suggest_float("lr", finder_cfg.min_lr, finder_cfg.max_lr, step=finder_cfg.step)
        if lr in self.param_history:
            # restore model before pruning
            self.restore_model()
            raise optuna.exceptions.TrialPruned()
        self.param_history.add(lr)
        for param_group in self.optims[self.main_model_name].param_groups:
            param_group["lr"] = round(lr,6)
        print(f"training with next lr: {lr}")


    def backup_model(self):
        print("backuping model...")
        model_device = next(self.models[self.main_model_name].parameters()).device
        # explicitly put the model on the CPU before storing it in memory
        self.state_cacher.store(key="model", state_dict=get_model_attr(self.models[self.main_model_name], 'cpu')().state_dict())
        self.state_cacher.store(key="optimizer", state_dict=self.optims[self.main_model_name].state_dict())
        # restore the model device
        get_model_attr(self.models[self.main_model_name],'to')(model_device)

    def restore_model(self):
        print("restoring model and seeds to initial state...")
        model_device = next(self.models[self.main_model_name].parameters()).device
        get_model_attr(self.models[self.main_model_name], 'load_state_dict')(self.state_cacher.retrieve("model"))
        self.optims[self.main_model_name].load_state_dict(self.state_cacher.retrieve("optimizer"))
        get_model_attr(self.models[self.main_model_name],'to')(model_device)
        set_random_seed(self.seed)

    def train(self, print_freq=10, fixbase_epoch=0, open_layers=None, lr_finder=False, perf_monitor=None,
              stop_callback=None):
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        accuracy = AverageMeter()

        self.set_model_mode('train')

        if not self._should_freeze_aux_models(self.epoch):
            # NB: it should be done before `two_stepped_transfer_learning`
            # to give possibility to freeze some layers in the unlikely event
            # that `two_stepped_transfer_learning` is used together with nncf
            self._unfreeze_aux_models()

        self.two_stepped_transfer_learning(
            self.epoch, fixbase_epoch, open_layers
        )

        if self._should_freeze_aux_models(self.epoch):
            self._freeze_aux_models()

        self.num_batches = len(self.train_loader)
        end = time.time()
        for self.batch_idx, data in enumerate(self.train_loader):
            if perf_monitor and not lr_finder: perf_monitor.on_train_batch_begin()

            data_time.update(time.time() - end)

            if self.compression_ctrl:
                self.compression_ctrl.scheduler.step(self.batch_idx)

            loss_summary, avg_acc = self.forward_backward(data)
            batch_time.update(time.time() - end)

            losses.update(loss_summary)
            accuracy.update(avg_acc)
            if perf_monitor and not lr_finder: perf_monitor.on_train_batch_end()

            if not lr_finder and (((self.batch_idx + 1) % print_freq) == 0 or
                                        self.batch_idx == self.num_batches - 1):
                nb_this_epoch = self.num_batches - (self.batch_idx + 1)
                nb_future_epochs = (self.max_epoch - (self.epoch + 1)) * self.num_batches
                eta_seconds = batch_time.avg * (nb_this_epoch+nb_future_epochs)
                eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
                print(
                    'epoch: [{0}/{1}][{2}/{3}]\t'
                    'time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'cls acc {accuracy.val:.3f} ({accuracy.avg:.3f})\t'
                    'eta {eta}\t'
                    '{losses}\t'
                    'lr {lr:.6f}'.format(
                        self.epoch + 1,
                        self.max_epoch,
                        self.batch_idx + 1,
                        self.num_batches,
                        batch_time=batch_time,
                        data_time=data_time,
                        accuracy=accuracy,
                        eta=eta_str,
                        losses=losses,
                        lr=self.get_current_lr()
                    )
                )

            if self.writer is not None and not lr_finder:
                n_iter = self.epoch * self.num_batches + self.batch_idx
                self.writer.add_scalar('Train/time', batch_time.avg, n_iter)
                self.writer.add_scalar('Train/data', data_time.avg, n_iter)
                self.writer.add_scalar('Aux/lr', self.get_current_lr(), n_iter)
                self.writer.add_scalar('Accuracy/train', accuracy.avg, n_iter)
                for name, meter in losses.meters.items():
                    self.writer.add_scalar('Loss/' + name, meter.avg, n_iter)

            end = time.time()
            self.current_lr = self.get_current_lr()
            if stop_callback and stop_callback.check_stop():
                break
            if not lr_finder and self.use_ema_decay:
                self.ema_model.update(self.models[self.main_model_name])
            if self.per_batch_annealing:
                self.update_lr()

        return losses.meters['loss'].avg

    def forward_backward(self, data):
        raise NotImplementedError

    def test(
        self,
        epoch,
        dist_metric='euclidean',
        normalize_feature=False,
        visrank=False,
        visrank_topk=10,
        save_dir='',
        use_metric_cuhk03=False,
        ranks=(1, 5, 10, 20),
        rerank=False,
        lr_finder = False,
        test_only=False
    ):
        r"""Tests model on target datasets.

        .. note::

            This function has been called in ``run()``.

        .. note::

            The test pipeline implemented in this function suits both image- and
            video-reid. In general, a subclass of Engine only needs to re-implement
            ``extract_features()`` and ``parse_data_for_eval()`` (most of the time),
            but not a must. Please refer to the source code for more details.
        """

        self.set_model_mode('eval')
        targets = list(self.test_loader.keys())
        top1, cur_top1, ema_top1 = [-1]*3
        should_save_ema_model = False

        for dataset_name in targets:
            domain = 'source' if dataset_name in self.datamanager.sources else 'target'
            print('##### Evaluating {} ({}) #####'.format(dataset_name, domain))
            # TO DO reduce amount of code for evaluation functions (DRY rule)
            for model_id, (model_name, model) in enumerate(self.models.items()):
                ema_condition = (self.use_ema_decay and not lr_finder
                        and not test_only and model_name == self.main_model_name)
                model_type = get_model_attr(model, 'type')
                if model_type == 'classification':
                    # do not evaluate second model till last epoch
                    if (model_name != self.main_model_name
                        and not test_only and epoch != (self.max_epoch - 1)):
                        continue
                    cur_top1 = self._evaluate_classification(
                        model=model,
                        epoch=epoch,
                        data_loader=self.test_loader[dataset_name]['query'],
                        model_name=model_name,
                        dataset_name=dataset_name,
                        ranks=ranks,
                        lr_finder=lr_finder
                    )
                    if ema_condition:
                        ema_top1 = self._evaluate_classification(
                            model=self.ema_model.module,
                            epoch=epoch,
                            data_loader=self.test_loader[dataset_name]['query'],
                            model_name='EMA model',
                            dataset_name=dataset_name,
                            ranks=ranks,
                            lr_finder = lr_finder
                        )
                elif model_type == 'contrastive':
                    pass
                elif model_type == 'multilabel':
                    # do not evaluate second model till last epoch
                    if (model_name != self.main_model_name
                        and not test_only and epoch != (self.max_epoch - 1)):
                        continue
                    # we compute mAP, but consider it top1 for consistency
                    # with single label classification
                    cur_top1 = self._evaluate_multilabel_classification(
                        model=model,
                        epoch=epoch,
                        data_loader=self.test_loader[dataset_name]['query'],
                        model_name=model_name,
                        dataset_name=dataset_name,
                        lr_finder=lr_finder
                    )
                    if ema_condition:
                        ema_top1 = self._evaluate_multilabel_classification(
                            model=self.ema_model.module,
                            epoch=epoch,
                            data_loader=self.test_loader[dataset_name]['query'],
                            model_name='EMA model',
                            dataset_name=dataset_name,
                            lr_finder = lr_finder
                        )
                elif dataset_name == 'lfw':
                    self._evaluate_pairwise(
                        model=model,
                        epoch=epoch,
                        data_loader=self.test_loader[dataset_name]['pairs'],
                        model_name=model_name
                    )
                else:
                    cur_top1 = self._evaluate_reid(
                        model=model,
                        epoch=epoch,
                        model_name=model_name,
                        dataset_name=dataset_name,
                        query_loader=self.test_loader[dataset_name]['query'],
                        gallery_loader=self.test_loader[dataset_name]['gallery'],
                        dist_metric=dist_metric,
                        normalize_feature=normalize_feature,
                        visrank=visrank,
                        visrank_topk=visrank_topk,
                        save_dir=save_dir,
                        use_metric_cuhk03=use_metric_cuhk03,
                        ranks=ranks,
                        rerank=rerank,
                        lr_finder=lr_finder
                    )

                if model_id == 0:
                    # the function should return accuracy results for the first (main) model only
                    if self.use_ema_decay and ema_top1 >= cur_top1:
                        should_save_ema_model = True
                        top1 = ema_top1
                    else:
                        top1 = cur_top1

        return top1, should_save_ema_model

    @torch.no_grad()
    def _evaluate_multilabel_classification(self, model, epoch, data_loader, model_name, dataset_name, lr_finder):
        mAP, mean_p_c, mean_r_c, mean_f_c, p_o, r_o, f_o = metrics.evaluate_multilabel_classification(data_loader, model, self.use_gpu)

        if self.writer is not None and not lr_finder:
            self.writer.add_scalar('Val/{}/{}/mAP'.format(dataset_name, model_name), mAP, epoch + 1)

        if not lr_finder:
            print('** Results ({}) **'.format(model_name))
            print('mAP: {:.2%}'.format(mAP))
            print('P_O: {:.2%}'.format(p_o))
            print('R_O: {:.2%}'.format(r_o))
            print('F_O: {:.2%}'.format(f_o))
            print('mean_P_C: {:.2%}'.format(mean_p_c))
            print('mean_R_C: {:.2%}'.format(mean_r_c))
            print('mean_F_C: {:.2%}'.format(mean_f_c))

        return mAP

    @torch.no_grad()
    def _evaluate_classification(self, model, epoch, data_loader, model_name, dataset_name, ranks, lr_finder):
        labelmap = []

        if data_loader.dataset.classes and get_model_attr(model, 'classification_classes') and \
                len(data_loader.dataset.classes) < len(get_model_attr(model, 'classification_classes')):

            for class_name in sorted(data_loader.dataset.classes.keys()):
                labelmap.append(data_loader.dataset.classes[class_name])

        cmc, mAP, norm_cm = metrics.evaluate_classification(data_loader, model, self.use_gpu, ranks, labelmap)

        if self.writer is not None and not lr_finder:
            self.writer.add_scalar('Val/{}/{}/mAP'.format(dataset_name, model_name), mAP, epoch + 1)
            for i, r in enumerate(ranks):
                self.writer.add_scalar('Val/{}/{}/Rank-{}'.format(dataset_name, model_name, r), cmc[i], epoch + 1)

        if not lr_finder:
            print('** Results ({}) **'.format(model_name))
            print('mAP: {:.2%}'.format(mAP))
            for i, r in enumerate(ranks):
                print('Rank-{:<3}: {:.2%}'.format(r, cmc[i]))
            if norm_cm.shape[0] <= 20:
                metrics.show_confusion_matrix(norm_cm)

        return cmc[0]

    @torch.no_grad()
    def _evaluate_pairwise(self, model, epoch, data_loader, model_name):
        same_acc, diff_acc, overall_acc, auc, avg_optimal_thresh = metrics.evaluate_lfw(
            data_loader, model, verbose=False
        )

        if self.writer is not None:
            self.writer.add_scalar('Val/LFW/{}/same_accuracy'.format(model_name), same_acc, epoch + 1)
            self.writer.add_scalar('Val/LFW/{}/diff_accuracy'.format(model_name), diff_acc, epoch + 1)
            self.writer.add_scalar('Val/LFW/{}/accuracy'.format(model_name), overall_acc, epoch + 1)
            self.writer.add_scalar('Val/LFW/{}/AUC'.format(model_name), auc, epoch + 1)

        print('\n** Results ({}) **'.format(model_name))
        print('Accuracy: {:.2%}'.format(overall_acc))
        print('Accuracy on positive pairs: {:.2%}'.format(same_acc))
        print('Accuracy on negative pairs: {:.2%}'.format(diff_acc))
        print('Average threshold: {:.2}'.format(avg_optimal_thresh))

    @torch.no_grad()
    def _evaluate_reid(
        self,
        model,
        epoch,
        dataset_name='',
        query_loader=None,
        gallery_loader=None,
        dist_metric='euclidean',
        normalize_feature=False,
        visrank=False,
        visrank_topk=10,
        save_dir='',
        use_metric_cuhk03=False,
        ranks=(1, 5, 10, 20),
        rerank=False,
        model_name='',
        lr_finder=False
    ):
        def _feature_extraction(data_loader):
            f_, pids_, camids_ = [], [], []
            for _, data in enumerate(data_loader):
                imgs, pids, camids = self.parse_data_for_eval(data)
                if self.use_gpu:
                    imgs = imgs.cuda()

                features = model(imgs),
                features = features.data.cpu()

                f_.append(features)
                pids_.extend(pids)
                camids_.extend(camids)

            f_ = torch.cat(f_, 0)
            pids_ = np.asarray(pids_)
            camids_ = np.asarray(camids_)

            return f_, pids_, camids_

        qf, q_pids, q_camids = _feature_extraction(query_loader)
        gf, g_pids, g_camids = _feature_extraction(gallery_loader)

        if normalize_feature:
            qf = F.normalize(qf, p=2, dim=1)
            gf = F.normalize(gf, p=2, dim=1)

        distmat = metrics.compute_distance_matrix(qf, gf, dist_metric)
        distmat = distmat.numpy()

        if rerank:
            distmat_qq = metrics.compute_distance_matrix(qf, qf, dist_metric)
            distmat_gg = metrics.compute_distance_matrix(gf, gf, dist_metric)
            distmat = re_ranking(distmat, distmat_qq, distmat_gg)

        cmc, mAP = metrics.evaluate_rank(
            distmat,
            q_pids,
            g_pids,
            q_camids,
            g_camids,
            use_metric_cuhk03=use_metric_cuhk03
        )

        if self.writer is not None and not lr_finder:
            self.writer.add_scalar('Val/{}/{}/mAP'.format(dataset_name, model_name), mAP, epoch + 1)
            for r in ranks:
                self.writer.add_scalar('Val/{}/{}/Rank-{}'.format(dataset_name, model_name, r), cmc[r - 1], epoch + 1)
        if not lr_finder:
            print('** Results ({}) **'.format(model_name))
            print('mAP: {:.2%}'.format(mAP))
            print('CMC curve')
            for r in ranks:
                print('Rank-{:<3}: {:.2%}'.format(r, cmc[r - 1]))

        if visrank and not lr_finder:
            visualize_ranked_results(
                distmat,
                self.datamanager.fetch_test_loaders(dataset_name),
                self.datamanager.data_type,
                width=self.datamanager.width,
                height=self.datamanager.height,
                save_dir=osp.join(save_dir, 'visrank_' + dataset_name),
                topk=visrank_topk
            )

        return cmc[0]

    @staticmethod
    def compute_loss(criterion, outputs, targets, **kwargs):
        if isinstance(outputs, (tuple, list)):
            loss = DeepSupervision(criterion, outputs, targets, **kwargs)
        else:
            loss = criterion(outputs, targets, **kwargs)
        return loss

    @staticmethod
    def parse_data_for_train(data, output_dict=False, enable_masks=False, use_gpu=False):
        imgs = data[0]

        obj_ids = data[1]
        if use_gpu:
            imgs = imgs.cuda()
            obj_ids = obj_ids.cuda()

        if output_dict:
            if len(data) > 3:
                dataset_ids = data[3].cuda() if use_gpu else data[3]

                masks = None
                if enable_masks:
                    masks = data[4].cuda() if use_gpu else data[4]

                attr = [record.cuda() if use_gpu else record for record in data[5:]]
                if len(attr) == 0:
                    attr = None
            else:
                dataset_ids = torch.zeros_like(obj_ids)
                masks = None
                attr = None

            return dict(img=imgs, obj_id=obj_ids, dataset_id=dataset_ids, mask=masks, attr=attr)
        else:
            return imgs, obj_ids

    @staticmethod
    def parse_data_for_eval(data):
        imgs = data[0]
        obj_ids = data[1]
        cam_ids = data[2]

        return imgs, obj_ids, cam_ids

    def two_stepped_transfer_learning(self, epoch, fixbase_epoch, open_layers):
        """Two-stepped transfer learning.

        The idea is to freeze base layers for a certain number of epochs
        and then open all layers for training.

        Reference: https://arxiv.org/abs/1611.05244
        """

        if (epoch + 1) <= fixbase_epoch and open_layers is not None:
            print('* Only train {} (epoch: {}/{})'.format(open_layers, epoch + 1, fixbase_epoch))

            for model in self.models.values():
                open_specified_layers(model, open_layers, strict=False)
        else:
            for model in self.models.values():
                open_all_layers(model)
