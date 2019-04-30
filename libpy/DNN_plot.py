from matplotlib import pyplot
import numpy


def plot_history(history, metric='mean_absolute_error'):
    pyplot.figure()
    pyplot.xlabel('Epoch')
    pyplot.ylabel('Mean Abs Error [1000$]')

    val_metric = 'val_' + metric
  
    pyplot.plot(history.epoch, numpy.array(history.history[metric]), label='Train Loss')
    pyplot.plot(history.epoch, numpy.array(history.history[val_metric]), label = 'Val loss')
    
    pyplot.legend()
    # pyplot.ylim([0, 5])

    pyplot.grid(True)

    pyplot.show()


def plot_predict(test_labels, test_predictions):
    pyplot.scatter(test_labels, test_predictions)
    
    pyplot.xlabel('True Values [1000$]')
    pyplot.ylabel('Predictions [1000$]')
    
    pyplot.axis('equal')
    
    pyplot.xlim(pyplot.xlim())
    pyplot.ylim(pyplot.ylim())
    
    _ = pyplot.plot([-100, 100], [-100, 100])

    pyplot.grid(True)

    pyplot.show()



def plot_predict_error(test_labels, test_predictions):
    error = test_predictions - test_labels
    
    pyplot.hist(error, bins = 50)
    
    pyplot.xlabel("Prediction Error [1000$]")
    _ = pyplot.ylabel("Count")

    pyplot.grid(True)

    pyplot.show()



def plot_compare_history(histories, key='mean_absolute_error'):
    pyplot.figure(figsize=(16,10))
    
    for name, history in histories:
        val = pyplot.plot(history.epoch, history.history['val_'+key],
                   '--', label=name.title()+' Val')
                
        pyplot.plot(history.epoch, history.history[key], color=val[0].get_color(),
             label=name.title()+' Train')

    pyplot.xlabel('Epochs')
    pyplot.ylabel(key.replace('_',' ').title())

    pyplot.grid(True)
    
    pyplot.legend()

    # pyplot.xlim([0,max(history.epoch)])

    pyplot.show()