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
        Diagnostics.Debug.Assert(input.Shape.Rank = 1) // Input should be 1D vector (or tensor).
        let inputDim = input.Shape.[0] // Get dimension of input.

        let s = [| outputDim; inputDim |] // Dimension data for NDShape to be used in TIMES operation. Each value at `i`th index represents size of `i`th dimension.
        // For example, if inputDim is 3 and outputDim is 4, `s` represents the size of (4 x 3).

        // Creates parameter for TIMES operation, left-side.
        // Size is `s`(hence 4 by 3 matrix), data type is `Float`,
        // Initializer is Glorot-Uniform-Initializer()
        let timesParam = new Parameter(
                            NDShape.CreateNDShape(s), DataType.Float, CNTKLib.GlorotUniformInitializer(
                                // (float)CNTKLib.DefaultParamInitScale, CNTKLib.SentinelValueForInferParamInitRank, CNTKLib.SentinelValueForInferParamInitRank, 1u
                                1.0, 3, 3, 1u
                            ), device, "timesParam"
                        ) // DataType.Float has been used since .NET API for CNTK does not support other data types.
        // NOTE: Parameter initializer takes following flow.
        // 1. Developer calls some initializer from CNTKLib.
        //      - https://github.com/Microsoft/CNTK/blob/987b22a8350211cb4c44278951857af1289c3666/Source/CNTKv2LibraryDll/Variable.cpp#L288
        // 2. Each initializer takes appropriate parameters, and for initializer types that require filter rank and output rank,
        //    to calculate random init values(mainly used for getting fan-in, fan-out.), they are required.
        //      - https://github.com/Microsoft/CNTK/blob/624bf7d82b341863a282c416110df71a0b3ea302/Source/ComputationNetworkLib/InputAndParamNodes.cpp#L240
        // 3. When the parameter value is used for the first time, if not initialized, parameter values will be initialized with given initializer.
        //      - https://github.com/Microsoft/CNTK/blob/987b22a8350211cb4c44278951857af1289c3666/Source/CNTKv2LibraryDll/Variable.cpp#L119
        //
        // For example - MLP,
        // Input: 784 (Think MNIST)
        // Filter: 1 (Since MLP just times weight and pluses bias.)
        // Output: 200 (Think single hidden layer with 200 percentrons.)
        // In this case, filterRank will be `1`, outputRank will be `1`.
        // And for Glorot-Uniform-Initialization,
        // fan-in will be `1 * 200 = numberOfFilterElements * numberOfInputElements = 200`
        // fan-out will be `1 * 784 * 200 * 1 / 200 = numberOfElements * numberOfFilterElements / fan-in = 784`.
        // 
        // For example - convolution,
        // Input: 28 * 28 * 3 (Think MNIST as color)
        // Filter: 3 * 3 * 3 (Kernel width, height is 3, depth is same as input's.)
        // Output: 13 * 13 * 12 (Think filter stride of 2, feature map count as 12.)
        // In this case, filterRank will be `3`, outputRank will be `3`.
        // And for Glorot-Uniform-Initialization,
        // fan-in will be `(3 * 3 * 3) * (28 * 28 * 3) = numberOfFilterElements * numberOfInputElements = 63504`,
        // fan-out will be `((3 * 3 * 3) * (28 * 28 * 3) * (13 * 13 * 12)) * (3 * 3 * 3) / ((3 * 3 * 3) * (28 * 28 * 3)) = numberOfElements * numberOfFilterElements / fan-in = 54756`.
        //
        // For details about input, output, kernel(filter): https://github.com/Microsoft/CNTK/blob/c78f40d0c1245fd4e6ffa09a3c943937b9feca74/Source/ComputationNetworkLib/ConvolutionalNodes.h

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
