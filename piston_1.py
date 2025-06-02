import numpy as np
import matplotlib.pyplot as plt
from waveoptics.tensors.numpy import crop_2d



class DMDPiston:
    WIDTH = 768#1024
    HEIGHT = 768
    # roi_shifts: tuple[int, int] = (384, 512)
 

    def __init__(self,
                 n_act_1d: int = 8,
                 roi_size: int = 450,
                 roi_shifts: tuple[int, int] = (384, 384)) -> None:
        self.n_act_1d = n_act_1d
        self.n_act_2d = self.n_act_1d ** 2
        self.size_act = None
        self.roi_size = roi_size
        self.roi_centers_xy = roi_shifts
        self.reduced_field_vector = None
        self.reduced_field_matrix = None
        self.field_matrix = None
        self.actus=None

    def generate_fields(self, n: int, rand_amp: bool = False, rand_phi: bool = False, phi: np.ndarray = None, phi_clx: np.ndarray = None) -> None:
        """
        Generate complex fields with random amplitude and phase.
        
        Args:
            n (int): Taille des champs générés (n x n).
            rand_amp (bool): Si True, génère une amplitude aléatoire. Sinon, utilise une amplitude unitaire.
            rand_phi (bool): Si True, génère une phase aléatoire. Sinon, utilise une phase nulle.
            phi (np.ndarray): Carte de phase personnalisée (optionnelle).
            phi_clx (np.ndarray): Champs complexes personnalisés (optionnels), de forme (..., 3).
        
        Raises:
            ValueError: Si la forme de 'phi' ne correspond pas à (n, n).
        """
        if phi_clx is not None:
            # Générer les champs complexes à partir de phi_clx
            phi_clx = phi_clx[..., 0] * (phi_clx[..., 1] + 1j * phi_clx[..., 2])
            phi = np.angle(phi_clx)
            amp = np.abs(phi_clx)
            field = amp * np.exp(1j * phi)
            print('calculated complex map from phi_clx ! ' )
            
        else:
            if phi is not None:
                # Vérifier la validité de la phase personnalisée
                if phi.shape != (n, n):
                    print(f"Le tableau 'phi' a une forme différente de ({n}, {n})")
                    plt.figure(figsize=(15,5))
                    plt.subplot(1,3,1)
                    plt.imshow(phi[...,0], cmap='jet')  
                    plt.colorbar(label='amp Normalised')
                    plt.title('Carte de phase (phase personnalisée)')
                    plt.subplot(1,3,2)
                    plt.imshow(phi[...,1], cmap='jet')  
                    plt.colorbar(label='cosine ')
                    plt.title('Carte de cos optimisés')
                    plt.subplot(1,3,3)
                    plt.imshow(phi[...,2], cmap='jet')  
                    plt.colorbar(label='sine ')
                    plt.title('Carte de sin optimisés')
                    # Construire la phase complexe
                    phii = phi[..., 1] + 1j *phi[...,2]
                    phii = np.angle(phii)
                    plt.figure()
                    plt.imshow(phii, cmap='jet')  
                    plt.colorbar(label='Phase (radians)')
                    plt.title('Carte de phase (phase personnalisée)')
                    field = np.exp(1j * phii)

            else:
                
                # Générer une phase et une amplitude aléatoires si aucune phase personnalisée n'est fournie
                phi = 2 * np.pi * np.random.rand(n, n) if rand_phi else np.zeros((n, n))
                amp = np.random.rand(n, n) if rand_amp else np.ones((n, n))
                # print('amp max:', np.max(amp))
                field = amp * np.exp(1j * phi)
        
        # Retourner ou stocker le champ généré selon vos besoins
        self.field = field                        
        self.actus=field
        self.reduced_field_matrix = field
        self.image_from_fields()

    def image_from_fields(self, field_array: np.ndarray = None) -> None:
        """Generate an image map from the complex field matrix."""

        field_array = field_array if field_array is not None else self.reduced_field_matrix
        if field_array is None:
            raise ValueError("Field array is not provided or generated.")

        n = int(np.sqrt(field_array.size))
        ideal_actu_size = int(np.ceil(self.roi_size / n))
        # print(f'ideal actuator size= {ideal_actu_size} ')
        self.size_act = ideal_actu_size

        field_map = np.repeat(np.repeat(field_array, ideal_actu_size, axis=0), ideal_actu_size, axis=1)

        if field_map.shape[0] != self.roi_size:
            print('field_map.shape[0] != self.roi_size, so, Crop2D ; field_map.shape[0]: ',field_map.shape[0])
            field_map = crop_2d(field_map, new_shape=(self.roi_size, self.roi_size))
            
        
        map = np.zeros((self.HEIGHT, self.WIDTH), dtype=np.complex64)
        map[:self.roi_size, :self.roi_size] = field_map

        map = np.roll(map, shift=self.roi_centers_xy[0] - self.roi_size // 2, axis=0)
        map = np.roll(map, shift=self.roi_centers_xy[1] - self.roi_size // 2, axis=1)
        self.field_matrix = map
        # self._visualize_field_map(map)

    def _visualize_field_map(self, field_map: np.ndarray) -> None:
        """Visualize the field map's phase and amplitude."""
        plt.figure(figsize=(12, 6))
        
        # Amplitude visualization
        plt.subplot(1, 2, 1)
        plt.title("np.abs(field_map)")
        plt.imshow(np.abs(field_map), cmap='gray')
        plt.colorbar()
        
        # Phase visualization
        plt.subplot(1, 2, 2)
        plt.title("np.angle(field_map)")
        plt.imshow(np.angle(field_map), cmap='gray')
        plt.colorbar()
        
        plt.tight_layout()
        plt.show()
