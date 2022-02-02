import Cocoa
import CreateML

let data = try MLDataTable(contentsOf: URL(fileURLWithPath:"/Users/enduser/Desktop/Projects/TwitStock/twitter-sanders-apple3.csv"))

let(trainingData,testingData) = data.randomSplit(by: 0.8, seed: 5)

let sentimentClassifier = try MLTextClassifier(trainingData: trainingData, textColumn: "text", labelColumn: "class")

let evalMet = sentimentClassifier.evaluation(on: testingData, textColumn: "text", labelColumn: "class")

let acc = (1.0-evalMet.classificationError)*100

print(evalMet)
let metaData = MLModelMetadata(author: "Rajat M. Jain", shortDescription: "Sentimental analysis on Tweets", version: "1.0")

try sentimentClassifier.write(to: URL(fileURLWithPath:"/Users/enduser/Desktop/Projects/TwitStock/TweetSentimentalClassifierModel.mlmodel"), metadata: metaData)

