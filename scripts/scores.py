from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, f1_score, cohen_kappa_score
from keras.metrics import MeanIoU
import numpy as np


class Score:

    def __init__(self, model, path_to_weights, X_test, Y_test, n_classes, Y_test_cat):
        self.model = model
        self.X_test = X_test
        self.Y_test = Y_test
        self.n_classes = n_classes
        self.Y_test_cat = Y_test_cat
        self.path_to_weights = path_to_weights
        self.check_()

    def check_(self):
        try:
            self.model.load_weights(self.path_to_weights)
            print(f'Model loaded: {self.path_to_weights}')
        except Exception as err:
            print(err)
            quit()

    # define a function to print classification report with test data
    def cs_report(self):
        y_pred = self.model.predict(self.X_test)
        y_pred_argmax = np.argmax(y_pred, axis=3)
        y_flat = self.Y_test.reshape(-1)
        y_pred_falt = y_pred_argmax.reshape(-1)
        print(classification_report(y_flat, y_pred_falt, digits=4))
        return y_pred_argmax

    def mean_iou(self, y_pred_argmax):
        n_classes = self.n_classes
        IOU_keras = MeanIoU(num_classes=n_classes)
        IOU_keras.update_state(self.Y_test, y_pred_argmax)
        print("\nMean IoU =", IOU_keras.result().numpy())
        return IOU_keras

    def iou_classes(self, IOU_keras):
        # To calculate I0U for each class...

        values = np.array(IOU_keras.get_weights()).reshape(self.n_classes, self.n_classes)
        if self.n_classes == 4:
            class1_IoU = values[0, 0] / (
                    values[0, 0] + values[0, 1] + values[1, 0] + values[0, 3] + values[0, 2] + values[2, 0] + values[
                3, 0])
            class2_IoU = values[1, 1] / (
                    values[1, 1] + values[1, 0] + values[0, 1] + values[1, 3] + values[1, 2] + values[2, 1] + values[
                3, 1])
            class3_IoU = values[2, 2] / (
                    values[2, 2] + values[2, 0] + values[2, 1] + values[2, 3] + values[0, 2] + values[1, 2] + values[
                3, 2])
            class4_IoU = values[3, 3] / (
                    values[3, 3] + values[3, 0] + values[3, 1] + values[3, 2] + values[0, 3] + values[1, 3] + values[
                2, 3])
            print("\nIoU for class1 is: ", class1_IoU)
            print("IoU for class2 is: ", class2_IoU)
            print("IoU for class3 is: ", class3_IoU)
            print("IoU for class4 is: ", class4_IoU)
        elif self.n_classes == 2:
            class1_IoU = values[0, 0] / (values[0, 0] + values[0, 1] + values[1, 0])
            class2_IoU = values[1, 1] / (values[1, 1] + values[1, 0] + values[0, 1])
            print("\nIoU for class1 is: ", class1_IoU)
            print("IoU for class2 is: ", class2_IoU)
        else:
            print(f'Metrics for {self.n_classes} classes not available.')

    def accuracy(self):
        _, dice, acc = self.model.evaluate(self.X_test, self.Y_test_cat)
        print("\nAccuracy is = ", (acc * 100.0), "%")

    def calc_scores(self):
        print('Calculating Scores...')
        y_pred_max = self.cs_report()
        IOU_keras = self.mean_iou(y_pred_max)
        self.iou_classes(IOU_keras)
        self.accuracy()
