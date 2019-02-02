namespace CNTKExamples

open System
open System.Collections.Generic
open System.Linq
open CNTK
open System.Runtime.InteropServices
open System.ComponentModel
open CNTK
open System.IO
open System.Collections.Generic
open System.Collections.Generic

type Activation = None = 0 | ReLU = 1 | Sigmoid = 2 | Tanh = 3

[<AbstractClassAttribute; SealedAttribute>]
type TestCommon =
    member this.TestDataDirPrefix = ""

type TestHelper() =
    static member FullyConnectedLinearLayer (input: Variable) (outputDim: int) (device: DeviceDescriptor) ([<DefaultParameterValueAttribute("")>]outputName: string) =
        Diagnostics.Debug.Assert(input.Shape.Rank = 1)
        let inputDim = input.Shape.[0]

        let s = [| outputDim; inputDim |]
        let timesParam = new Parameter(
                            NDShape.CreateNDShape(s), DataType.Float, CNTKLib.GlorotUniformInitializer(
                                (float)CNTKLib.DefaultParamInitScale, CNTKLib.SentinelValueForInferParamInitRank, CNTKLib.SentinelValueForInferParamInitRank, 1u
                            ), device, "timesParam"
                        ) // DataType.Float has been used since .NET API for CNTK does not support other data types.
        let timesFunction = CNTKLib.Times(timesParam, input, "times")

        let s2 = [| outputDim |]
        let plusParam = new Parameter(NDShape.CreateNDShape(s2), 0.0f, device, "plusParam") // Float32 has been used since .NET API for CNTK does not support other data types.
        CNTKLib.Plus(plusParam, new Variable(timesFunction), outputName)

    static member Dense
        (input: Function)
        (outputDim: int)
        (device: DeviceDescriptor)
        ([<DefaultParameterValueAttribute(Activation.None)>]activation: Activation)
        ([<DefaultParameterValueAttribute("")>]outputName: string) =
            // This code may not consider the case when input is Variable, i.e. having Shape property.
            // See https://github.com/Microsoft/CNTK/blob/master/Examples/TrainingCSharp/Common/TestHelper.cs#L30
            let mutable inputFunction = input
            let inputFunctionAsVariable = new Variable(inputFunction)
            if (inputFunctionAsVariable.Shape.Rank <> 1) then
                let newDim = inputFunctionAsVariable.Shape.Dimensions.Aggregate(fun d1 d2 -> d1 * d2)
                inputFunction <- CNTKLib.Reshape(inputFunctionAsVariable, NDShape.CreateNDShape([| newDim |]))
            let fullyConnected = TestHelper.FullyConnectedLinearLayer (new Variable(inputFunction)) outputDim device outputName
            match activation with
            | Activation.None ->
                fullyConnected
            | Activation.ReLU ->
                CNTKLib.ReLU (new Variable(fullyConnected))
            | Activation.Sigmoid ->
                CNTKLib.Sigmoid (new Variable(fullyConnected))
            | Activation.Tanh ->
                CNTKLib.Tanh (new Variable(fullyConnected))
            | _ ->
                raise (InvalidEnumArgumentException("activation", (int)activation, typeof<Activation>))
    
    static member ValidateModelWithMinibatchSource
        (modelFile: string)
        (testMinibatchSource: MinibatchSource)
        (featureInputName: string)
        (labelInputName: string)
        (outputName: string)
        (device: DeviceDescriptor)
        ([<DefaultParameterValueAttribute(1000)>]maxCount: int) =
            let model = Function.Load(modelFile, device)
            let imageInput = model.Arguments.[0]
            let labelOutput = model.Outputs.Single(fun o -> o.Name = outputName)

            let featureStreamInfo = testMinibatchSource.StreamInfo featureInputName
            let labelStreamInfo = testMinibatchSource.StreamInfo labelInputName

            let batchSize = 50
            let mutable misCountTotal, totalCount = (0, 0)

            let mutable brk = false
            while not brk do
                let minibatchData = testMinibatchSource.GetNextMinibatch((uint32)batchSize, device)
                if minibatchData |> isNull || minibatchData.Count = 0 then
                    brk <- true
                else
                    totalCount <- totalCount + (int)minibatchData.[featureStreamInfo].numberOfSamples

                    let labelData = minibatchData.[labelStreamInfo].data.GetDenseData<float32>(labelOutput) // Float32 has been used since .NET API for CNTK does not support other data types.
                    let expectedLabels = labelData.Select(fun l -> l.IndexOf(l.Max())).ToList()

                    let inputDataMap = new Dictionary<Variable, Value>()
                    inputDataMap.Add(imageInput, minibatchData.[featureStreamInfo].data)

                    let outputDataMap = new Dictionary<Variable, Value>()
                    outputDataMap.Add(labelOutput, null)

                    model.Evaluate(inputDataMap, outputDataMap, device)
                    let outputData = outputDataMap.[labelOutput].GetDenseData<float32>(labelOutput) // Float32 has been used since .NET API for CNTK does not support other data types.
                    let actualLabels = outputData.Select(fun l -> l.IndexOf(l.Max())).ToList()

                    let misMatches = actualLabels.Zip(expectedLabels, fun a b -> if a.Equals(b) then 0 else 1).Sum()

                    misCountTotal <- misCountTotal + misMatches
                    Console.WriteLine("Validating Model: Total Samples = {0}, Misclassify Count = {1}", totalCount, misCountTotal)

                    if totalCount > maxCount then
                        brk <- true

            let errorRate = 1.0F * (float32)misCountTotal / (float32)totalCount
            Console.WriteLine("Model Validation Error = {0}", errorRate)
            errorRate

    static member SaveAndReloadModel
        (func: Function ref)
        (variables: IList<Variable>)
        (device: DeviceDescriptor)
        ([<DefaultParameterValueAttribute(0)>]rank: uint32) =
            let tempModelPath = "feedForward.net" + rank.ToString()
            File.Delete(tempModelPath)

            let inputVariableUids = new Dictionary<string, Variable>()
            let outputVariableNames = new Dictionary<string, Variable>()

            for variable in variables do
                if variable.IsOutput then
                    outputVariableNames.Add(variable.Owner.Name, variable)
                else
                    inputVariableUids.Add(variable.Uid, variable)
            
            func.Value.Save(tempModelPath)
            func.Value <- Function.Load(tempModelPath, device)

            File.Delete(tempModelPath)

            let inputs = func.Value.Inputs
            for inputVariableInfo in inputVariableUids.ToList() do
                let newInputVariable = inputs.First(fun v -> v.Uid = inputVariableInfo.Key)
                inputVariableUids.[inputVariableInfo.Key] <- newInputVariable
            
            let outputs = func.Value.Outputs
            for outputVariableInfo in outputVariableNames.ToList() do
                let newOutputVariable = outputs.First(fun v -> v.Owner.Name = outputVariableInfo.Key)
                outputVariableNames.[outputVariableInfo.Key] <- newOutputVariable

    static member MiniBatchDataIsSweepEnd (minibatchValues: ICollection<MinibatchData>) =
        minibatchValues.Any(fun a -> a.sweepEnd)
       
    static member PrintTrainingProgress (trainer: Trainer) (minibatchIdx: int) (outputFrequencyInMinibatches: int) =
        if (minibatchIdx % outputFrequencyInMinibatches) = 0 && trainer.PreviousMinibatchSampleCount() <> 0u then
            let trainLossValue = (float)(trainer.PreviousMinibatchEvaluationAverage())
            let evaluationValue = (float)(trainer.PreviousMinibatchLossAverage())
            Console.WriteLine("Minibatch: {0} CrossEntropyLoss = {1}, EvaluationCriterion = {2}", minibatchIdx, trainLossValue, evaluationValue)

    static member PrintOutputDims (func: Function) (functionName: string) =
        let shape = func.Output.Shape

        if shape.Rank = 3 then
            Console.WriteLine("{0} dim0: {1}, dim1: {2}, dim2: {3}", functionName, shape.[0], shape.[1], shape.[2])
        else
            Console.WriteLine("{0} dim0: {1}", functionName, shape.[0])
