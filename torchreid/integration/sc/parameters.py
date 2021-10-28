
from attr import attrs
from sys import maxsize

from ote_sdk.configuration.elements import (ParameterGroup,
                                            add_parameter_group,
                                            boolean_attribute,
                                            configurable_integer,
                                            selectable,
                                            string_attribute,
                                            configurable_boolean
                                            )
from ote_sdk.configuration.configurable_parameters import ConfigurableParameters
from ote_sdk.configuration.model_lifecycle import ModelLifecycle

from .parameters_enums import POTQuantizationPreset

@attrs
class OTEClassificationParameters(ConfigurableParameters):
    header = string_attribute("Configuration for an object detection task")
    description = header

    @attrs
    class __LearningParameters(ParameterGroup):
        header = string_attribute("Learning Parameters")
        description = header

        batch_size = configurable_integer(
            default_value=32,
            min_value=1,
            max_value=512,
            header="Batch size",
            description="The number of training samples seen in each iteration of training. Increasing this value "
            "improves training time and may make the training more stable. A larger batch size has higher "
            "memory requirements.",
            warning="Increasing this value may cause the system to use more memory than available, "
            "potentially causing out of memory errors, please update with caution.",
            affects_outcome_of=ModelLifecycle.TRAINING
        )

        max_num_epochs = configurable_integer(
            default_value=200,
            min_value=1,
            max_value=100000,
            header="Maximum number of training epochs",
            description="Increasing this value causes the results to be more robust but training time "
            "will be longer.",
            affects_outcome_of=ModelLifecycle.TRAINING
        )

    @attrs
    class __AlgoBackend(ParameterGroup):
        header = string_attribute("Internal Algo Backend parameters")
        description = header
        visible_in_ui = boolean_attribute(False)

        template = string_attribute("template.yaml")
        model = string_attribute("main_model.yaml")
        multilabel_model = string_attribute("main_model_multilabel.yaml")
        model_name = string_attribute("image classification model")

    @attrs
    class __POTParameter(ParameterGroup):
        header = string_attribute("POT Parameters")
        description = header

        stat_subset_size = configurable_integer(
            header="Number of data samples",
            description="Number of data samples used for post-training optimization",
            default_value=300,
            min_value=1,
            max_value=maxsize
        )

        preset = selectable(default_value=POTQuantizationPreset.PERFORMANCE, header="Preset",
                            description="Quantization preset that defines quantization scheme",
                            editable=False, visible_in_ui=False)

    @attrs
    class __DebugParameters(ParameterGroup):
        header = string_attribute("Debugging Parameters")
        description = header

        enable_debug_dump = configurable_boolean(
            default_value=True,
            header="Enable data dumps for debugging",
            description="Enable data dumps for debugging",
            affects_outcome_of=ModelLifecycle.NONE
        )

    learning_parameters = add_parameter_group(__LearningParameters)
    algo_backend = add_parameter_group(__AlgoBackend)
    pot_parameters = add_parameter_group(__POTParameter)
    debug_parameters = add_parameter_group(__DebugParameters)
