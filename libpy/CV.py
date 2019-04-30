# Cross_Validation

import numpy, pandas
from sklearn.model_selection import train_test_split

from matplotlib import pyplot

def validate_model(model, X, y, test_size=0.4, n=100):
    scores = []
    for i in range(n):
        # distributing Training and Testing set
        X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=test_size )
        print('.', end='')
        model.fit(X_train, y_train)
        print('.', end='')
        scores.append(model.score(X_test, y_test))
        print('. ', end='') 
    print(" ... validate_model complete")
    scores = numpy.array(scores)
    
    return sum(scores) / len(scores), model


def grade_model(ESTIMATORS, X_train, y_train, X_test, y_test) :
    predictions = dict()

    for name, estimator in ESTIMATORS.items():    
        print(name, " : ", estimator)

        score, estimator = validate_model( estimator, X_train, y_train )

        print("score: ", score)

        ESTIMATORS[name] = estimator

        y_pred = estimator.predict( X_test ).flatten()

        prediction = []
        prediction.append( y_test.values.flatten() )
        prediction.append(y_pred)
        predictions[name] = prediction 

        plot_predict_error(y_test, y_pred, name)

    return ESTIMATORS, predictions

def plot_predict(predictions):
    
    
    #     # figure, ax = pyplot.subplots()
    #     for label in predictions :
    #         pyplot.scatter( predictions[label][0], predictions[label][1], label=label )
        
    #     pyplot.xlabel('True Values [1000$]')
    #     pyplot.ylabel('Predictions [1000$]')
        
    #     pyplot.axis('equal')
        
    #     pyplot.xlim(pyplot.xlim())
    #     pyplot.ylim(pyplot.ylim())
        
    #     #_ = pyplot.plot([-100, 100], [-100, 100])
        
    #     pyplot.legend(loc='best')
    #     pyplot.grid(True)

    #     pyplot.show()
    i = 331   
    pyplot.figure(figsize=(20,20))
    for label in predictions :        
        
        pyplot.subplot(i)   
        pyplot.scatter( predictions[label][0], predictions[label][1], label=label )
    
        pyplot.xlabel('True Values [1000$]')
        pyplot.ylabel('Predictions [1000$]')

        pyplot.title(label)

        pyplot.axis('equal')

        pyplot.xlim(pyplot.xlim())
        pyplot.ylim(pyplot.ylim())

        #_ = pyplot.plot([-100, 100], [-100, 100])
    
        pyplot.legend()
        pyplot.grid(True)
        
        i += 1

    pyplot.show()


def plot_predict_error(test_labels, test_predictions, label=""):
    error = test_predictions - test_labels
    
    pyplot.hist(error, bins = 50)
    
    pyplot.xlabel("Prediction Error [1000$]")
    _ = pyplot.ylabel("Count")

    pyplot.title(label)

    pyplot.show()


def pack_data(predictions):
    
    df = pandas.DataFrame()
    for label, value in predictions.items():
        
        # print(label)
        y_test, y_pred = value[0], value[1]

        column1 = "y_test"

        if column1 not in df.columns:
            dummy_df = pandas.DataFrame( { column1: y_test } )
            df = pandas.concat( [ df, dummy_df ], axis=1 )

        column2 = "".join([ label, "_", "y_pred" ])

        dummy_df = pandas.DataFrame( { column2:numpy.subtract(y_test, y_pred) } )
        # print( dummy_df )
        df = pandas.concat( [ df, dummy_df ], axis=1 )
        # print (df)

    return df



def plot_any( x, y, xlabel='', ylabel='' ):

    pyplot.scatter(x, y)

    pyplot.xlabel(xlabel)
    pyplot.ylabel(ylabel)
    
    pyplot.show()