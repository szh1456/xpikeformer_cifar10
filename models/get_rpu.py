from aihwkit.simulator.configs import (
    SingleRPUConfig,
    InferenceRPUConfig,
    FloatingPointRPUConfig,
    ConstantStepDevice,
    FloatingPointDevice,
    WeightNoiseType,
    WeightClipType,
    WeightModifierType,
    TorchInferenceRPUConfig,
    WeightRemapType
)
from aihwkit.simulator.parameters.io import IOParameters
from aihwkit.inference import PCMLikeNoiseModel, GlobalDriftCompensation
from aihwkit.simulator.presets.inference import StandardHWATrainingPreset
def get_rpu(rpu='pcm'):
    # rpu config from examples
    rpu_config = InferenceRPUConfig()
    
    rpu_config.forward = IOParameters()
    rpu_config.forward.is_perfect = False
    rpu_config.forward.inp_res = 1/64.  # 6-bit DAC discretization.
    rpu_config.forward.out_res = 1/64. # 6-bit ADC discretization.
    rpu_config.forward.w_noise_type = WeightNoiseType.ADDITIVE_CONSTANT
    rpu_config.forward.w_noise = 0.02  # Short-term w-noise.
    rpu_config.forward.out_noise = 0.02 # Some output noise.

    rpu_config.clip.type = WeightClipType.FIXED_VALUE
    rpu_config.clip.fixed_value = 1.0

    rpu_config.mapping.weight_scaling_omega = 1.0
    rpu_config.mapping.weight_scaling_columnwise = True
    rpu_config.mapping.learn_out_scaling = True
    rpu_config.mapping.out_scaling_columnwise = True

    rpu_config.modifier.pdrop = 0.03  # Drop connect.
    rpu_config.modifier.type = WeightModifierType.ADD_NORMAL  # Fwd/bwd weight noise.
    rpu_config.modifier.std_dev = 0.1
    rpu_config.modifier.rel_to_actual_wmax = True
    rpu_config.modifier.per_batch_sample = False

    rpu_config.remap.type = WeightRemapType.LAYERWISE_SYMMETRIC # this cause significant change

    rpu_config.pre_post.input_range.enable = True

    
    rpu_config.noise_model = PCMLikeNoiseModel(g_max=25.0)
    if rpu == 'pcm':
        rpu_config.drift_compensation = GlobalDriftCompensation()
    elif rpu == 'baseline':
        rpu_config.drift_compensation = None
    return rpu_config

if __name__ == '__main__':
    rpu = get_rpu()
    print(rpu)
    rpu = get_rpu('baseline')
    print(rpu)