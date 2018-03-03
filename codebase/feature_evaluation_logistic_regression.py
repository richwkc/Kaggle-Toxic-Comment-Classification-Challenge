from evaluate_predictions import evaluatePredictions

from sklearn.linear_model import LogisticRegression


def trainModel(features, labels):
    return LogisticRegression().fit(features, labels)


def getPredictions(model, features):
    return model.predict(features), model.predict_proba(features)


def evaluateFeaturesWithLogisticRegression(trainFeatures, testFeatures, trainLabels, testLabels):
    model = trainModel(trainFeatures, trainLabels)

    trainPredictions, trainProbPredictions = getPredictions(model, trainFeatures)
    testPredictions, testProbPredictions = getPredictions(model, testFeatures)

    evaluatePredictions(testLabels, testPredictions, testProbPredictions)
    evaluatePredictions(trainLabels, trainPredictions, trainProbPredictions)
