import numpy as np
import matplotlib.pyplot as plt
import config as cf
from proc_data import load_data, train_validation_split, generate_batch

#
# Visualization of distribution of steering angles in source dataset.
#
def view_distribution_of_steering_angles(data, title, bins=100):
    data_steering = np.float32(np.array(data)[:, cf.STEERING])
    
    plt.title(title)
    plt.hist(data_steering, bins, normed=0, facecolor='blue')
    plt.ylabel('number of frames')
    plt.xlabel('steering angle')
    plt.tick_params(axis='both', which='major', labelsize=8)
    plt.show()
    print('min steering:', np.min(data_steering))
    print('msx steering:', np.max(data_steering))


#
# Visualization of ground truth distribution depending on biasing towards non-0 value.
# 
def view_steering_depending_on_bias_parameter(train_data):

    biases = np.linspace(start=0., stop=1., num=5)
    
    fig, axArray = plt.subplots(len(biases))
    
    plt.suptitle('Steering angle distribution depending on bias parameter', fontsize=14, fontweight='bold')
    
    for i, ax in enumerate(axArray.ravel()):
        b = biases[i]
        im_batch, steer_batch = generate_batch(train_data, batch_size=2048, augment=True, bias=b)
        ax.hist(steer_batch, 100, normed=1, facecolor='blue')
        ax.set_title('Bias: {:02f}'.format(b))
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.axis([-1., 1., 0., 2.])
        
    plt.tight_layout(pad=2, w_pad=0.5, h_pad=1.0)
    plt.show()

    
def explore_input_data():
    data= load_data(cf.DRIVING_LOG)
    view_distribution_of_steering_angles(data, 'Steering angle distribution in sample data')
    
#    data_train, data_val = train_validation_split(data)
#    distribution_of_steering_angles(data_train, 'Steering angle distribution in training data')
#    distribution_of_steering_angles(data_val, 'Steering angle distribution in validating data')
    
if __name__ == '__main__':

#    explore_input_data()
    data= load_data(cf.DRIVING_LOG)
    view_steering_depending_on_bias_parameter(data)