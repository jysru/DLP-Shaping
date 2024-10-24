import numpy as np
from waveoptics.tensors.numpy import crop_2d
import cv2

_DEFAULT_SCREEN_WIDTH = 1680
_DEFAULT_SCREEN_HEIGHT = 1050

class DMDPiston:
    WIDTH = 1024
    HEIGHT = 768
    _screen_width: int = _DEFAULT_SCREEN_WIDTH
    _screen_height: int = _DEFAULT_SCREEN_HEIGHT

    def __init__(self,
                 n_act_1d: int = 8,
                 roi_size: int = 192,
                 roi_shifts: tuple[int] = (384, 512),
                 ) -> None:
        self.n_act_1d = n_act_1d
        self.n_act_2d = self.n_act_1d ** 2
        self.size_act = None
        self.roi_size = roi_size
        self.roi_centers_xy = roi_shifts
        self.reduced_field_vector = None
        self.reduced_field_matrix = None
        self.field_matrix = None

    def generate_fields(self, n: int, rand_amp: bool = True, rand_phi: bool = True) -> np.ndarray:
        phi = 2 * np.pi * np.random.rand(n, n)
        amp = np.random.rand(n, n)
        if rand_amp and rand_phi:
            field = amp * np.exp(1j * phi)
        elif rand_amp and not rand_phi:
            field = amp * np.exp(1j * np.zeros_like(phi))
        elif not rand_amp and rand_phi:
            field = np.ones_like(amp) * np.exp(1j * phi)
        else:
            field = np.ones_like(amp) * np.exp(1j * np.zeros_like(phi))
        self.reduced_field_matrix = np.copy(field)
        self.image_from_fields(self.reduced_field_matrix)

    def image_from_fields(self, field_array: np.ndarray = None):
        field_array = field_array if field_array is not None else self.reduced_field_matrix
        self.reduced_field_vector = field_array.flatten()
        n = np.round(np.sqrt(field_array.size)).astype(np.int32)
        
        ideal_actu_size = int(np.ceil(self.roi_size / n))
        self.size_act = ideal_actu_size

        field_map = np.repeat(field_array, repeats=ideal_actu_size, axis=0)
        field_map = np.repeat(field_map, repeats=ideal_actu_size, axis=1)

        if field_map.shape[0] != self.roi_size: 
            field_map = crop_2d(field_map, new_shape=(self.roi_size, self.roi_size))

        map = np.zeros((self.HEIGHT, self.WIDTH), dtype=np.complex64)

        map[0:self.roi_size, 0:self.roi_size] = field_map
        map = np.roll(map, shift=self.roi_centers_xy[0] - self.roi_size // 2, axis=0)
        map = np.roll(map, shift=self.roi_centers_xy[1] - self.roi_size // 2, axis=1)

        self.field_matrix = map
