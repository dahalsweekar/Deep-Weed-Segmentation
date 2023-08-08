import random
import numpy as np
import matplotlib.pyplot as plt


class Test:
    def __init__(self, model, weight_path, X_test, Y_test):
        self.model = model
        self.weight_path = weight_path
        self.X_test = X_test
        self.Y_test = Y_test
        self.check_()

    def check_(self):
        try:
            self.model.load_weights(self.weight_path)
            print(f'Model loaded: {self.weight_path}')
        except Exception as err:
            print(err)
            quit()

    def test(self):
        print('Testing on a random image...')
        test_img_number = random.randint(0, len(self.X_test))
        test_img = self.X_test[test_img_number]
        ground_truth = self.Y_test[test_img_number]
        # test_img_norm=test_img[:,:,0][:,:,None]
        test_img_input = np.expand_dims(test_img, 0)
        prediction = (self.model.predict(test_img_input))
        predicted_img = np.argmax(prediction, axis=3)[0, :, :]

        plt.figure(figsize=(12, 8))
        plt.subplot(231)
        plt.title('Testing Image')
        plt.imshow(test_img)
        plt.subplot(232)
        plt.title('Testing Label')
        plt.imshow(ground_truth, cmap='jet')
        plt.subplot(233)
        plt.title('Prediction on test image')
        plt.imshow(predicted_img, cmap='jet')
        plt.savefig('./plots/figure.png')
        print('Figure Saved.')
        plt.show()
