namespace CNTKExamples

open System
open System.Collections.Generic
open System.Linq
open CNTK
open CNTKExamples

type LogisticRegression() =
    static member inputDim = 3
    static member numOutputClasses = 2

    static member private CreateLinearModel (input: Variable) (outputDim: int) (device: DeviceDescriptor) =
        let inputDim = input.Shape.[0]
        let weightParam = new Parameter(NDShape.CreateNDShape([| outputDim; inputDim |]), DataType.Float, 1.0, device, "w") // DataType.Float has been used since .NET API for CNTK does not support other data types.
        let biasParam = new Parameter(NDShape.CreateNDShape([| outputDim |]), DataType.Float, 0.0, device, "b") // DataType.Float has been used since .NET API for CNTK does not support other data types.
        CNTKLib.Plus(new Variable(CNTKLib.Times(weightParam, input)), biasParam)

    static member private GenerateGaussianNoise (mean: float32) (stdDev: float32) (random: Random) =
        let u1 = 1.0 - random.NextDouble()
        let u2 = 1.0 - random.NextDouble()
        let stdNormalRandomValue = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2)
        mean + stdDev + (float32)stdNormalRandomValue

    static member private GenerateRawDataSamples (sampleSize: int) (inputDim: int) (numOutputClasses: int) =
        let random = Random(0)
        
        let features = new List<float32>(Array.init (sampleSize * inputDim) (fun _ -> 0.0f)) // Float32 has been used since .NET API for CNTK does not support other data types.
        let oneHotLabels = new List<float32>(Array.init (sampleSize * numOutputClasses) (fun _ -> 0.0f)) // Float32 has been used since .NET API for CNTK does not support other data types.

        for sample in 0..(sampleSize - 1) do
            let label = random.Next(numOutputClasses)
            for i in 0..(numOutputClasses - 1) do
                oneHotLabels.[sample * numOutputClasses + i] <- if label = i then 1.0f else 0.0f
            for i in 0..(numOutputClasses - 1) do
                features.[sample * inputDim + i] <- (LogisticRegression.GenerateGaussianNoise 3.0f 1.0f random)
        
        (features, oneHotLabels)

    static member private GenerateValueData (sampleSize: int) (inputDim: int) (numOutputClasses: int) (device: DeviceDescriptor) =
        // let features: float List ref = ref (new List<float32>())
        // let oneHotLabels: float List ref = ref (new List<float32>())
        // LogisticRegression.GenerateRawDataSamples sampleSize inputDim numOutputClasses features oneHotLabels
        let features, oneHotLabels = LogisticRegression.GenerateRawDataSamples sampleSize inputDim numOutputClasses

        let featureValue = Value.CreateBatch<float32>(NDShape.CreateNDShape([| inputDim |]), features, device) // Float32 has been used since .NET API for CNTK does not support other data types.
        let labelValue = Value.CreateBatch<float32>(NDShape.CreateNDShape([| numOutputClasses |]), oneHotLabels, device) // Float32 has been used since .NET API for CNTK does not support other data types.

        (featureValue, labelValue)

    static member TrainAndEvaluate (device: DeviceDescriptor) =
        let featureVariable = Variable.InputVariable(NDShape.CreateNDShape([| LogisticRegression.inputDim |]), DataType.Float) // DataType.Float has been used since .NET API for CNTK does not support other data types.
        let labelVariable = Variable.InputVariable(NDShape.CreateNDShape([| LogisticRegression.numOutputClasses |]), DataType.Float) // DataType.Float has been used since .NET API for CNTK does not support other data types.
        let classifierOutput = LogisticRegression.CreateLinearModel featureVariable LogisticRegression.numOutputClasses device
        let loss = CNTKLib.CrossEntropyWithSoftmax(new Variable(classifierOutput), labelVariable)
        let evalError = CNTKLib.ClassificationError(new Variable(classifierOutput), labelVariable)

        let learningRatePerSample = new CNTK.TrainingParameterScheduleDouble(0.02, 1u)
        let parameterLearners = [| Learner.SGDLearner(classifierOutput.Parameters(), learningRatePerSample) |]
        let trainer = Trainer.CreateTrainer(classifierOutput, loss, evalError, parameterLearners)

        let minibatchSize = 64
        let numMinibatchesToTrain = 1000
        let updatePerMinibatches = 50

        for minibatchCount in 0..(numMinibatchesToTrain - 1) do
         
            let features, labels = LogisticRegression.GenerateValueData minibatchSize LogisticRegression.inputDim LogisticRegression.numOutputClasses device
            let trainDictionary = new Dictionary<Variable, Value>()
            trainDictionary.Add(featureVariable, features)
            trainDictionary.Add(labelVariable, labels)
            trainer.TrainMinibatch(trainDictionary, true, device) |> ignore
            TestHelper.PrintTrainingProgress trainer minibatchCount updatePerMinibatches

        let testSize = 100    
        let testFeatureValue, expectedLabelValue = LogisticRegression.GenerateValueData testSize LogisticRegression.inputDim LogisticRegression.numOutputClasses device

        let expectedOneHot = expectedLabelValue.GetDenseData<float32>(labelVariable) // Float32 has been used since .NET API for CNTK does not support other data types.
        let expectedLabels = expectedOneHot.Select(fun l -> l.IndexOf(1.0f)).ToList()

        let inputDataMap = new Dictionary<Variable, Value>()
        inputDataMap.Add(featureVariable, testFeatureValue)
        let outputDataMap = new Dictionary<Variable, Value>()
        outputDataMap.Add(classifierOutput.Output, null)
        classifierOutput.Evaluate(inputDataMap, outputDataMap, device)
        let outputValue = outputDataMap.[classifierOutput.Output]
        let actualLabelSoftMax = outputValue.GetDenseData<float32>(classifierOutput.Output) // Float32 has been used since .NET API for CNTK does not support other data types.
        let actualLabels = actualLabelSoftMax.Select(fun l -> l.IndexOf(l.Max())).ToList()
        let misMatches = actualLabels.Zip(expectedLabels, fun a b -> if a.Equals(b) then 0 else 1).Sum()

        Console.WriteLine("Validating Model: Total samples = {0}, Misclassify Count = {1}", testSize, misMatches)