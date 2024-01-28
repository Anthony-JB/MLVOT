import numpy as np
class KalmanFilter:
    def __init__(self, dt, u_x, u_y, std_acc, x_sdt_meas, y_sdt_meas):
        """
        Initialise les paramètres de la classe KalmanFilter.

        Args:
            dt: le temps pour un cycle utilisé pour estimer l'état (temps d'échantillonnage)
            u_x: l'accélération dans la direction $x$
            u_y: l'accélération dans la direction $y$
            std_acc: l'amplitude du bruit de processus
            x_sdt_meas: l'écart type des mesures dans la direction $x$
            y_sdt_meas: l'écart type des mesures dans la direction $y$

        Returns:
            None
        """

        self.dt = dt
        self.u = np.array([u_x, u_y])
        self.std_acc = std_acc
        self.x_sdt_meas = x_sdt_meas
        self.y_sdt_meas = y_sdt_meas

        # Définissez l'état initial

        self.x = np.array([0, 0, 0, 0])

        # Définissez les matrices A et B

        self.A = np.array([[1, 0, dt, 0], 
                           [0, 1, 0, dt], 
                           [0, 0, 1, 0], 
                           [0, 0, 0, 1]])
        self.B = np.array([[0.5 * dt ** 2, 0], 
                           [0, 0.5 * dt ** 2], 
                           [dt, 0], 
                           [dt, 0]])

        # Définissez la matrice H

        self.H = np.array([[1, 0, 0, 0], 
                           [0, 1, 0, 0]])

        # Définissez la matrice Q

        self.Q = np.array([[0.25 * dt ** 4, 0, 0.5 * dt ** 3, 0],
                           [0, 0.25 * dt ** 4, 0, 0.5 * dt ** 3],
                           [0.5 * dt ** 3, 0, dt ** 2, 0],
                           [0, 0.5 * dt ** 3, 0, dt ** 2]]) * std_acc ** 2

        # Définissez la matrice R

        self.R = np.array([[x_sdt_meas, 0], 
                           [0, y_sdt_meas]])

        # Définissez la matrice P pour l'erreur de prédiction comme une matrice d'identité dont la forme est la même que la forme de la matrice A.

        self.P = np.identity(4)
    
    def predict(self):
        """
        Prédit l'état en fonction de l'état actuel et des entrées de contrôle.

        Returns:
            None
        """

        # Prédire l'état
        x_pred = self.A @ self.x + self.B @ self.u


        # Prédire l'erreur de prédiction

        P_pred = self.A @ self.P @ self.A.T + self.Q

        return x_pred, P_pred
    
    def update(self, z):
        """
        Mise à jour de l'état estimé en fonction des nouvelles mesures.

        Args:
            z: les nouvelles mesures

        Returns:
            None
        """

        x_pred, P_pred = self.predict()

        # Calculez le gain de Kalman

        K = P_pred @ self.H.T @ np.linalg.inv(self.H @ P_pred @ self.H.T + self.R)

        # Mettez à jour l'état estimé
        self.x = x_pred + K @ (z - self.H @ x_pred)[0]

        # Mettez à jour l'erreur de prédiction

        self.P = (np.eye(4) - K @ self.H) @ P_pred

        return None

