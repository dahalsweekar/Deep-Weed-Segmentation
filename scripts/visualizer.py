import matplotlib.pyplot as plt


class Visualizer:

    def __init__(self, results):
        self.results = results

    def plot_curve(self):
        loss = self.results.history['loss']
        val_loss = self.results.history['val_loss']
        epochs = range(1, len(loss) + 1)
        plt.plot(epochs, loss, 'y', label='Training loss')
        plt.plot(epochs, val_loss, 'r', label='Validation loss')
        plt.title('Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('./plots/loss.png')
        plt.show()

        acc = self.results.history['iou_score']
        val_acc = self.results.history['val_iou_score']
        plt.plot(epochs, acc, 'y', label='Training IOU_Score')
        plt.plot(epochs, val_acc, 'r', label='Validation IOU Score')
        plt.title('Training and validation IoU')
        plt.xlabel('Epochs')
        plt.ylabel('IoU')
        plt.legend()
        plt.savefig('./plots/accuracy.png')
        plt.show()
