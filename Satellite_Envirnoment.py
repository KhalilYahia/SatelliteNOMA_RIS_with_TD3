import gym
from gym import spaces
import numpy as np
from scipy.special import jn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Rotation matrix around x axis
def Rotat_x(a):
    return np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])

class LEO_RIS_NOMA_Env(gym.Env):
    def __init__(self, num_users=2, ris_elements=16, max_power=10.0, freq=2e9, K_factor=10, time_step=0,
                 antenna_gain=30):
        super(LEO_RIS_NOMA_Env, self).__init__()

        # Constants
        self.c = 3e8  # Speed of light (m/s)
        self.freq = freq  # Frequency (Hz)
        self.K_factor = K_factor  # Rician K-factor
        self.time_step = time_step  # Time step in seconds
        self.k = 1.38e-23  # Boltzmann constant
        self.T = 290  # Noise temperature in Kelvin
        self.B = 20e6  # Bandwidth in Hz (20 MHz)
        self.Noise = self.k * self.T * self.B
        self.transmitted_power = 10 # 10 W

        self.ris_elements = ris_elements # number of RIS elements

        self.lambda_wave= self.c /self.freq # wavelength
        self.ris_element_width = self.lambda_wave/5
        self.ris_element_Max_Gain= (4*np.pi*self.ris_element_width**2)/(self.lambda_wave**2)

        # Maximum Antenna gain (in linear)
        self.maxantenna_gain = 10**(antenna_gain/10)  # Satellite antenna gain in dBi
        self.antenna_Diameter = 1.5 # in m
        self.maxantenna_gain = (np.pi*self.antenna_Diameter/self.lambda_wave)**2
        #print(10*np.log10(self.maxantenna_gain))
        self.angle_3_db =np.deg2rad(70*self.lambda_wave/self.antenna_Diameter)

        # Earth's radius and satellite parameters
        self.earth_radius = 6371e3  # Earth radius in meters
        self.leo_altitude = 1500e3  # LEO satellite altitude in meters
        self.leo_orbital_radius = self.earth_radius + self.leo_altitude  # Orbital radius
        self.leo_orbital_speed = 7116 # 7.8e3  # Approximate orbital speed of a LEO satellite in m/s
        self.leo_angle = -np.pi/2  # Initial angle of satellite position (radians)

        # Users and RIS positions are constant in the XYZ plane
        self.num_users = num_users
        self.user_positions = np.zeros((num_users, 3))  # XYZ positions of users
        self.ris_positions = np.zeros((ris_elements, 3))  # XYZ positions of RIS elements
        #self._generate_user_and_ris_positions(self.user_positions[0],self.user_positions[1])  # Initialize constant positions
        self.leo_position = np.zeros((1, 3))
        # this changed with change positions
        self.Sat_Ris_antenna_gains=0
        self.Ris_User1_antenna_gains=0
        self.Ris_User2_antenna_gains=0

        # Define action and observation space
        self.action_space = spaces.Box(
            low=np.array([0.0] * (num_users-1) + [-np.pi] * ris_elements),
            high=np.array([max_power] * (num_users-1) + [np.pi] * ris_elements),
            dtype=np.float32
        )

        # Observation space: Channel gains (FSPL and Rician), RIS phases
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(num_users*2 + ris_elements*2 + ris_elements*4,), dtype=np.float32
        )

        # Initialize the state (channel gains with FSPL and Rician fading)
        self.state = 0 # self._channel_Generate()

        self.user_channel_gains =0
        self.Sat_Ris_channel_gains=0
        self.Ris_User1_channel_gains=0
        self.Ris_User2_channel_gains=0

        # actions
        self.ris_phase_shifts = np.zeros((16, 1))
        self.user1_power = 0
        self.user2_power = 0

    # To calculate the distance between the point C and the line AB
    def distance_point_to_line(self, LinePoint_A, LinePoint_B, C):
        # Convert points A, B, C to NumPy arrays
        A = np.array(LinePoint_A)
        B = np.array(LinePoint_B)
        C = np.array(C)

        # Calculate vectors AB and AC
        AB = B - A
        AC = C - A

        # Cross product of AB and AC
        cross_product = np.cross(AB, AC)

        # Calculate the magnitude of the cross product
        cross_product_magnitude = np.linalg.norm(cross_product)

        # Calculate the magnitude of vector AB
        AB_magnitude = np.linalg.norm(AB)

        # The distance is the ratio of the magnitudes
        distance = cross_product_magnitude / AB_magnitude

        return distance
    def _calculate_fspl(self, d):
        """Calculate free-space path loss."""
        return (4 * np.pi * d * self.freq / self.c) ** 2

    def _rician_fading(self, d):
        """Calculate Rician fading for a given distance."""
        fspl = self._calculate_fspl(d)

        # LoS component
        h_los = np.sqrt(1/fspl)

        # NLoS component (Rayleigh fading)
        h_nlos = np.random.rayleigh(scale=np.sqrt(1 / fspl))
        # Combine the LoS and NLoS components with the Rician K-factor
        h = np.sqrt(self.K_factor / (self.K_factor + 1)) * h_los + \
            np.sqrt(1 / (self.K_factor + 1)) * h_nlos
        return h_los # h

    def _calculate_antenna_gain(self, user_position, satellite_position):
        # User position
        X_u, Y_u, Z_u = user_position
        X_s, Y_s, Z_s = satellite_position
        User_to_beamCenter= self.distance_point_to_line(satellite_position,[0,0,0],user_position);
        User_to_satellite_= np.linalg.norm(satellite_position-user_position);

        sat_redation_angle= np.arcsin(User_to_beamCenter/User_to_satellite_)

        Uj_user = 2.07123 * np.sin(sat_redation_angle)/np.sin(self.angle_3_db)

        epsilon = 1e-8  # Small value to prevent division by zero
        Uj_user_safe = np.maximum(np.abs(Uj_user), epsilon) # Ensure no division by zero

        gain_=self.maxantenna_gain * (((jn(1,Uj_user_safe)/(2*Uj_user_safe)) + (36*jn(3,Uj_user_safe)/(Uj_user_safe**3)))**2)
        #gain_ = self.maxantenna_gain * (np.sinc(sat_redation_angle)**2)
        pdlam_ =np.pi* self.antenna_Diameter/self.lambda_wave
        #gain_ = self.maxantenna_gain * ((np.sin(pdlam_ *np.sin(sat_redation_angle)))/(pdlam_*np.sin(sat_redation_angle)))**2

       # if sat_redation_angle==0:
       #     gain_ = self.maxantenna_gain
       # else:
       #     gain_ = self.maxantenna_gain * (2 / pdlam_) * (
       #                 jn(1, pdlam_ * np.sin(sat_redation_angle)) / np.sin(sat_redation_angle))
       # if gain_>self.maxantenna_gain/2:
       #     print(gain_)
        return np.abs(gain_)  # Assume a constant gain for simplicity

    # RIS elements Antenna gains
    def theta_RIS(self, p1, p2):  # polar
        return np.arccos((p2[2] - p1[2]) / np.linalg.norm(p2 - p1, axis=0))
    def RIS_element_Antenna_Gain(self, p1, p2):
        return self.ris_element_Max_Gain * self.theta_RIS(p1, p2)

    def _generate_user_and_ris_positions(self,users_lat_long_alti,RIS_lat_long_alti):
        """Generate constant 3D positions for users and RIS in XYZ coordinates."""
        # Users randomly distributed on the surface of the Earth
        for i in range(self.num_users):
            latitude = users_lat_long_alti[i,0]
            longitude = users_lat_long_alti[i,1]
            altitude = users_lat_long_alti[i,2]
            self.user_positions[i] = self._geodetic_to_cartesian(latitude, longitude, altitude)

        # Define a central position for the RIS
        ris_latitude = RIS_lat_long_alti[0]  # Central latitude
        ris_longitude = RIS_lat_long_alti[1]  # Central longitude
        ris_altitude = RIS_lat_long_alti[2]  # Altitude for all RIS elements (e.g., 50 meters)

        # Offsets for RIS elements (for example, 1 meter apart in x and y directions)
        offset_distance = self.ris_element_width  # Distance between RIS elements
        for i in range(self.ris_elements):
            # Calculate offsets based on the index
            x_offset = (i % 4) * offset_distance  # Arrange in a grid pattern (4 elements per row)
            y_offset = (i // 4) * offset_distance  # Move to the next row every 4 elements
            # Convert from geodetic to Cartesian coordinates
            self.ris_positions[i] = self._geodetic_to_cartesian(ris_latitude, ris_longitude, ris_altitude)
            # Add the offsets
            self.ris_positions[i][0] += x_offset  # X offset
            self.ris_positions[i][1] += y_offset  # Y offset

    def _geodetic_to_cartesian(self, lat, lon, alt):
        """Convert geodetic coordinates (lat, lon, alt) to Cartesian XYZ coordinates."""
        r = self.earth_radius + alt
        x = r * np.cos(lat) * np.cos(lon)
        y = r * np.cos(lat) * np.sin(lon)
        z = r * np.sin(lat)
        return np.array([x, y, z])
    def _Satellite_move(self,time_):
        self.time_step = time_
        self._update_leo_position()
    def _update_leo_position(self):
        """Update the LEO satellite position based on its orbital motion."""
        # Satellite moves in a circular orbit around the Earth
        Angular_speed_ = self.leo_orbital_speed / self.leo_orbital_radius
        self.leo_angle = (Angular_speed_) * self.time_step
        # Ensure angle wraps around after 2π (completing a full orbit)
        self.leo_angle = self.leo_angle % (2 * np.pi)
        #print('time: ',self.time_step)
        # New LEO satellite position in Cartesian coordinates (circular orbit)
        x = self.leo_orbital_radius * np.cos(self.leo_angle)
        y = self.leo_orbital_radius * np.sin(self.leo_angle)
        z = 0  # Assume satellite orbits in the equatorial plane for simplicity

        leo_position = np.array([x, y, z])
        # Apply antenna gain to each user
        self.antenna_gains = np.array([self._calculate_antenna_gain(userpositions, leo_position) for userpositions in
                                  self.user_positions])
        self.Sat_Ris_antenna_gains = np.array([self._calculate_antenna_gain(element, leo_position) for element in
                                          self.ris_positions])
        self.Ris_Sat_antenna_gains = np.array([self.RIS_element_Antenna_Gain(leo_position, element) for element in
                                          self.ris_positions])

        self.Ris_User1_antenna_gains = np.array(
            [self.RIS_element_Antenna_Gain(element, self.user_positions[0]) for element in
             self.ris_positions])
        self.Ris_User2_antenna_gains = np.array(
            [self.RIS_element_Antenna_Gain(element, self.user_positions[1]) for element in
             self.ris_positions])
        self.leo_position = leo_position
        #print('time step: ',  self.time_step, 'leo position: ',self.leo_position)
        return leo_position

    def _channel_Generate(self):
        # Update LEO satellite position
        leo_position = self.leo_position

        # Calculate distances from the LEO satellite to each user
        user_distances = np.linalg.norm(leo_position - self.user_positions, axis=1)
        #print('time: ',self.time_step,' , user_distances: ',user_distances[0] ,' , leo_position: ',leo_position)

        # Calculate channel gains for each user (FSPL and Rician fading)
        user_channel_gains = np.array([self._rician_fading(d) for d in user_distances])
        #print(np.log10(user_channel_gains))
        # Calculate distances from the LEO satellite to each element in RIS
        Sat_Ris_distances = np.array([np.linalg.norm(leo_position - element, axis=0) for element
                                      in self.ris_positions])
        Ris_User1_distances = np.array([np.linalg.norm(self.user_positions[0] - element, axis=0) for element
                                        in self.ris_positions])
        Ris_User2_distances = np.array([np.linalg.norm(self.user_positions[1] - element, axis=0) for element
                                        in self.ris_positions])

        # Calculate channel gains for each element in RIS (FSPL and Rician fading)
        Sat_Ris_channel_gains = np.array([self._rician_fading(d) for d in Sat_Ris_distances])
        Ris_User1_channel_gains = np.array([self._rician_fading(d) for d in Ris_User1_distances])
        Ris_User2_channel_gains = np.array([self._rician_fading(d) for d in Ris_User2_distances])

        self.user_channel_gains=user_channel_gains
        self.Sat_Ris_channel_gains = Sat_Ris_channel_gains
        self.Ris_User1_channel_gains = Ris_User1_channel_gains
        self.Ris_User2_channel_gains = Ris_User2_channel_gains

    def _calculate_SNR(self):
        """Generate the state with channel information for each user."""
        # Update LEO satellite position
        leo_position = self.leo_position

        # Calculate distances from the LEO satellite to each user
        #user_distances = np.linalg.norm(leo_position - self.user_positions, axis=1)

        # Calculate channel gains for each user (FSPL and Rician fading)
        #user_channel_gains = np.array([self._rician_fading(d) for d in user_distances])

        # Apply antenna gain to each user
        #antenna_gains = np.array([self._calculate_antenna_gain(userpositions,leo_position) for userpositions in
        #                          self.user_positions])

        direct_channel_gains = np.multiply(self.user_channel_gains , np.sqrt(self.antenna_gains))

        ris_phase_shifts = np.exp(self.ris_phase_shifts)

        # Calculate distances from the LEO satellite to each element in RIS
        #Sat_Ris_distances = np.array([np.linalg.norm(leo_position - element, axis=1) for element
        #                              in self.ris_positions ])
        #Ris_User1_distances = np.array([np.linalg.norm(self.user_positions[0] - element, axis=1) for element
        #                              in self.ris_positions ])
        #Ris_User2_distances = np.array([np.linalg.norm(self.user_positions[1] - element, axis=1) for element
        #                                in self.ris_positions])
        # Calculate channel gains for each element in RIS (FSPL and Rician fading)
        #Sat_Ris_channel_gains = np.array([self._rician_fading(d) for d in Sat_Ris_distances])
        #Ris_User1_channel_gains = np.array([self._rician_fading(d) for d in Ris_User1_distances])
        #Ris_User2_channel_gains = np.array([self._rician_fading(d) for d in Ris_User2_distances])

        # Apply antenna gain to each element in RIS
        #Sat_Ris_antenna_gains = np.array([self._calculate_antenna_gain(element, leo_position) for element in
        #                          self.ris_positions])
        #Ris_Sat_antenna_gains = np.array([self.RIS_element_Antenna_Gain(leo_position,element) for element in
        #                                  self.ris_positions])

        Sat_Ris_channel_gains = np.multiply(self.Sat_Ris_channel_gains, np.sqrt(self.Sat_Ris_antenna_gains))
        Sat_Ris_channel_gains = np.multiply(Sat_Ris_channel_gains, np.sqrt(self.Ris_Sat_antenna_gains))

        #Ris_User1_antenna_gains = np.array([self.RIS_element_Antenna_Gain(element,self.user_positions[0]) for element in
        #                                  self.ris_positions])
        #Ris_User2_antenna_gains = np.array([self.RIS_element_Antenna_Gain(element, self.user_positions[1]) for element in
        #     self.ris_positions])

        Phase_Sat_Ris_channel_gains = ris_phase_shifts * Sat_Ris_channel_gains

        Undirect_channel_User1_gains = self.Ris_User1_channel_gains * Phase_Sat_Ris_channel_gains
        Undirect_channel_User2_gains = self.Ris_User2_channel_gains * Phase_Sat_Ris_channel_gains

        Undirect_channel_User1_gains = np.multiply(Undirect_channel_User1_gains, np.sqrt(self.Ris_User1_antenna_gains))
        Undirect_channel_User2_gains = np.multiply(Undirect_channel_User2_gains, np.sqrt(self.Ris_User2_antenna_gains))

        Sum_Undirect_channel_User1_gains = np.sum(Undirect_channel_User1_gains)
        Sum_Undirect_channel_User2_gains = np.sum(Undirect_channel_User2_gains)

        channel_Received_ByUser1 = np.abs(Sum_Undirect_channel_User1_gains + direct_channel_gains[0])**2
        channel_Received_ByUser2 = np.abs(Sum_Undirect_channel_User2_gains + direct_channel_gains[1])**2


        SNR1 = channel_Received_ByUser1 * self.user1_power * self.transmitted_power/self.Noise
        SNR2 = channel_Received_ByUser2 * self.user2_power * self.transmitted_power/(self.user1_power *self.transmitted_power * channel_Received_ByUser2 +self.Noise)
        #SNR2 = channel_Received_ByUser2 * self.user2_power * self.transmitted_power / ( self.Noise)
        return SNR1,SNR2 , direct_channel_gains[0]**2,direct_channel_gains[1]**2

    def _convert_channel_to_state(self):
        # Extract real and imaginary parts
        real_user_channel_gains = np.real(self.user_channel_gains)
        imag_user_channel_gains = np.imag(self.user_channel_gains)

        real_Sat_Ris_channel_gains = np.real(self.Sat_Ris_channel_gains)
        imag_Sat_Ris_channel_gains = np.imag(self.Sat_Ris_channel_gains)

        real_Ris_User1_channel_gains = np.real(self.Ris_User1_channel_gains)
        imag_Ris_User1_channel_gains = np.imag(self.Ris_User1_channel_gains)

        real_Ris_User2_channel_gains = np.real(self.Ris_User2_channel_gains)
        imag_Ris_User2_channel_gains = np.imag(self.Ris_User2_channel_gains)

        # For 2D array (2x3) - concatenate real and imaginary parts
        user_channel_gains_combined = np.concatenate((real_user_channel_gains, imag_user_channel_gains), axis=-1)

        # For 1D arrays - concatenate real and imaginary parts
        Sat_Ris_channel_gains_combined = np.concatenate((real_Sat_Ris_channel_gains, imag_Sat_Ris_channel_gains))
        Ris_User1_channel_gains_combined = np.concatenate((real_Ris_User1_channel_gains, imag_Ris_User1_channel_gains))
        Ris_User2_channel_gains_combined = np.concatenate((real_Ris_User2_channel_gains, imag_Ris_User2_channel_gains))

        # Combine all arrays into one input array
        combined_input = np.concatenate((
            user_channel_gains_combined.flatten(),  # Flatten the 2D array
            Sat_Ris_channel_gains_combined,
            Ris_User1_channel_gains_combined,
            Ris_User2_channel_gains_combined
        ), axis=0)

        return combined_input
    def _calculate_channel_state(self):
        self._channel_Generate()
        return self._convert_channel_to_state()
    def reset(self):
        # Reset the environment to an initial state
        self.state = self._calculate_channel_state()
        return self.state

    def step(self, action):
        # Action is a combination of power allocation and RIS phase shifts
        power_allocation = np.abs(action[1])/10
        ris_phase_shifts = action[1:1+self.ris_elements]*0.14
        # ris_phase_shifts_real = action[self.num_users:self.num_users+self.ris_elements]
        # ris_phase_shifts_imag = action[self.num_users+self.ris_elements:]
        #ris_phase_shifts = ris_phase_shifts_real + (1j * ris_phase_shifts_imag)

        # Calculate the signal-to-noise ratio (SNR) or other metrics
        reward = self.calculate_reward(power_allocation, ris_phase_shifts)

        # Update the state (channel changes as the satellite moves)
        #self.state = self._calculate_channel_state()

        done = False  # Define the terminal condition
        return self.state, reward, done, {}

    def calculate_reward(self, power_allocation, ris_phase_shifts):
        self.ris_phase_shifts = ris_phase_shifts
        self.user1_power = power_allocation
        self.user2_power = 1 - power_allocation
        user1_SNR,user2_SNR,Direct_channel_1,Direct_channel_2 = self._calculate_SNR()

        #print('user1_SNR')
        #print(user1_SNR)
        user1_rate = np.log10(1+user1_SNR)
        user2_rate = np.log10(1 + user2_SNR)

        if user1_rate<1 or user2_rate<1:
            reward=-1


        #print(user1_rate)
        #print(user2_rate)
        reward = user1_rate + user2_rate
        return reward

    def render(self, mode='human'):
        pass

    def plot_positions(self):
        """Plot the satellite, users, and RIS elements in 3D space."""
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot users
        user_x = self.user_positions[:, 0]
        user_y = self.user_positions[:, 1]
        user_z = self.user_positions[:, 2]
        ax.scatter(user_x, user_y, user_z, color='blue', label='Users')

        # Plot RIS elements
        ris_x = self.ris_positions[:, 0]
        ris_y = self.ris_positions[:, 1]
        ris_z = self.ris_positions[:, 2]
        ax.scatter(ris_x, ris_y, ris_z, color='green', label='RIS Elements')

        # Plot satellite
        #satellite_x = self.leo_position[0]
        #satellite_y = self.leo_position[1]
        #satellite_z = self.leo_position[2]
        #ax.scatter(satellite_x, satellite_y, satellite_z, color='red', label='Satellite', s=100)

        # Set labels
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        ax.set_zlabel('Z (meters)')
        ax.set_title('Satellite, Users, and RIS Elements Positions')
        plt.draw()
        # Animation loop
        time_steps = 900
        for t in range(time_steps):
            self._Satellite_move((t-time_steps/2))  # Get satellite position at time t
            sat_pos = self.leo_position
            # Update satellite plot
            ax.scatter(sat_pos[0], sat_pos[1], sat_pos[2], color='orange', label='Satellite' if t == 0 else "")

            plt.draw()
            plt.pause(0.000001)
        # Display legend
        ax.legend()

        # Set equal scaling for all axes
        ax.set_box_aspect([1, 1, 1])  # Aspect ratio is 1:1:1

        plt.ioff()
        plt.show()

    def plot_SNR(self):

        fig, ax = plt.subplots()
        time_steps = 900
        # Set labels
        ax.set_xlabel('X (time second)')
        ax.set_ylabel('Y (SNR log)')
        ax.set_title('SNR ')

        plt.draw()
        plt.grid()
        SNR_1= np.zeros(time_steps)
        SNR_2 = np.zeros(time_steps)
        channel_1 = np.zeros(time_steps)
        channel_2 = np.zeros(time_steps)
        # Animation loop

        for t in range(time_steps):
            self._Satellite_move(t-time_steps/2)  # Get satellite position at time t
            self._channel_Generate()
            SNR_1[t], SNR_2[t], channel_1[t] ,channel_2[t] =10*np.log10(self._calculate_SNR())

        # Display legend
        #print(channel_1)
        t_=np.zeros(time_steps)
        for t in range(time_steps):
            t_[t]=t-time_steps/2
        ax.plot(t_,SNR_1)
        ax.plot(t_, SNR_2)
        ax.legend()
        plt.show()

# User and RIS positions (on Earth)
#users_pos = np.array([[0, 0, 0], [0.008, 0.01, 0]])  # 2 users on Earth's surface
#RIS_pos = np.array([1, 1, 50])                     # RIS on Earth's surface
# Assuming your environment is initialized as env
#env = LEO_RIS_NOMA_Env()
#env._generate_user_and_ris_positions(users_pos,RIS_pos)

##env.ris_positions = RIS_pos
##env.plot_positions()
#env.user1_power = 0.3
#env.user2_power =0.7
#env.plot_SNR()