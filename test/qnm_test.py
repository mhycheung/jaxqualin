from jaxqualin.qnmode import str_to_mode
import numpy as np

def test_mode_value(test_waveform_tuple, omega220r_test, omega220i_test):
    _, Mf, af, retro = test_waveform_tuple
    mode220_test = str_to_mode('2.2.0', Mf, af, retro = retro)
    assert np.isclose(float(mode220_test.omegar), omega220r_test)
    assert np.isclose(float(mode220_test.omegai), omega220i_test)

# def test_retro_modes(test_waveform_tuple, omega220r_test, omega220i_test):
#     _, Mf, af, retro = test_waveform_tuple
#     mode220_test = str_to_mode('2.2.0', Mf, af, retro = retro)
#     mode220_test_retro = str_to_mode('2.2.0', Mf, af, retro = not retro)
#     mode2n20_test = str_to_mode('2.-2.0', Mf, af, retro = retro)
#     mode2n20_test_retro = str_to_mode('2.-2.0', Mf, af, retro = not retro)
#     assert np.isclose(float(mode220_test.omegar), -float(mode2n20_test_retro.omegar))
#     assert np.isclose(float(mode220_test.omegai), float(mode2n20_test_retro.omegai))
#     assert np.isclose(float(mode2n20_test.omegar), -float(mode220_test_retro.omegar))
#     assert np.isclose(float(mode2n20_test.omegai), float(mode220_test_retro.omegai))
#     assert np.isclose(float(mode220_test.omegar), -float(mode220_test_retro.omegar))
#     assert np.isclose(float(mode220_test.omegai), float(mode220_test_retro.omegai))