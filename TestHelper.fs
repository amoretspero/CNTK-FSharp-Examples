namespace CNTKExamples

open System
open System.Collections.Generic
open System.Linq
open CNTK
open System.Runtime.InteropServices
open System.ComponentModel
open System.IO

/// **Description**  
/// Type of activation functions.
/// 
///   * `None` - No activation is applied.
///   * `ReLU` - Rectified Linear Unit function is applied.
///   * `Sigmoid` - Sigmoid function is applied.
///   * `Tanh` - Tanh function is applied.
/// 
type Activation = None = 0 | ReLU = 1 | Sigmoid = 2 | Tanh = 3

[<AbstractClassAttribute; SealedAttribute>]
type TestCommon =
    member this.TestDataDirPrefix = ""

type TestHelper() =
    
    /// **Description**  
    /// Creates Fully Connected layer from input.
    /// Since input to fully connected layer should be 1D vector, that shape is assumed.
    /// Glorot-Uniform initializer is used for weight parameter.
    ///
    /// **Parameters**
    ///   * `input` - parameter of type `Variable`. Indicates input variable to this fully connected layer.
    ///   * `outputDim` - parameter of type `int`. Indicates the output variable's dimension.
    ///   * `device` - parameter of type `DeviceDescriptor`. Device context to create this fully connected layer.
    ///   * `outputName` - parameter of type `string`. Name of output variable.
    ///
    /// **Output Type**
    ///   * `Function` - Fully connected layer as function. One can think of this function's output as output variable or layer of fully connected layer.
    ///
    /// **Exceptions**
    ///
    static member FullyConnectedLinearLayer (input: Variable) (outputDim: int) (device: DeviceDescriptor) ([<DefaultParameterValueAttribute("")>]outputName: string) =
        Diagnostics.Debug.Assert(input.Shape.Rank = 1) // Input should be 1D vector (or tensor).
        let inputDim = input.Shape.[0] // Get dimension of input.

        let s = [| outputDim; inputDim |] // Dimension data for NDShape to be used in TIMES operation. Each value at `i`th index represents size of `i`th dimension.
        // For example, if inputDim is 3 and outputDim is 4, `s` represents the size of (4 x 3).

        // Creates parameter for TIMES operation, left-side.
        // Size is `s`(hence 4 by 3 matrix for above example.), data type is `Float`,
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
        // Filter: 1 (Since MLP just sums `weight * input` and pluses bias, filter can be thought as 1D vector of weights for neuron_{i+1}.)
        // Output: 200 (Think single hidden layer with 200 percentrons.)
        // In this case, filterRank will be `1`, outputRank will be `1`.
        // And for Glorot-Uniform-Initialization,
        // fan-in will be `1 * 784 = numberOfFilterElements * numberOfInputElements = 784`
        // fan-out will be `1 * 784 * 200 * 1 / 784 = numberOfElements * numberOfFilterElements / fan-in = 200`.
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

        // Creates TIMES function with left operand of `timesParam` and right operand of `input`, with name "times".
        let timesFunction = CNTKLib.Times(timesParam, input, "times")

        // PLUS function will be used to provide bias for this fully connected layer.
        // So, for PLUS function, only output dimension of fully connected layer is needed, resulting in 1D parameter.
        // Bias is initialized with 0.0f.        
        let s2 = [| outputDim |]
        let plusParam = new Parameter(NDShape.CreateNDShape(s2), 0.0f, device, "plusParam") // Float32 has been used since .NET API for CNTK does not support other data types.

        // Returns PLUS function with its left operand of PLUS parameter,
        // right operand of output of TIMES function we created earlier.(Explicitly casted to Variable.)
        CNTKLib.Plus(plusParam, new Variable(timesFunction), outputName)

    
    /// **Description**  
    /// Creates Dense layer from input.
    /// Dense layer consists of fully connected layer with activation function
    /// applied after fully connected layer.
    ///
    /// **Parameters**
    ///   * `input` - parameter of type `Function`. Indicates input function, that generates input variable to this dense layer.
    ///   * `outputDim` - parameter of type `int`. Indicates output dimension for this dense layer.
    ///   * `device` - parameter of type `DeviceDescriptor`. Device to create this dense layer on.
    ///   * `activation` - parameter of type `Activation`. Activation function to apply after fully connected layer.
    ///   * `outputName` - parameter of type `string`. Name of output variable.
    ///
    /// **Output Type**
    ///   * `Function` - Dense layer as function. One can think of this function's output as output variable or layer of dense layer.
    ///
    /// **Exceptions**
    ///
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

            // When input function's output does not have shape of 1D vector,
            // Calculate aggregated dimension for that output, by multiplying all dimension's size,
            // then reshape input function to 1D vector shape with aggregated dimension.
            if (inputFunctionAsVariable.Shape.Rank <> 1) then
                let newDim = inputFunctionAsVariable.Shape.Dimensions.Aggregate(fun d1 d2 -> d1 * d2)
                inputFunction <- CNTKLib.Reshape(inputFunctionAsVariable, NDShape.CreateNDShape([| newDim |]))

            // Generates fully connected layer.
            let fullyConnected = TestHelper.FullyConnectedLinearLayer (new Variable(inputFunction)) outputDim device outputName

            // Apply activation function to output of fully connected layer.
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
    
    
    /// **Description**  
    /// Validates model with minibatch.
    ///
    /// **Parameters**
    ///   * `modelFile` - parameter of type `string`. File name of model.
    ///   * `testMinibatchSource` - parameter of type `MinibatchSource`. Minibatch source to use for test.
    ///   * `featureInputName` - parameter of type `string`. Name of feature input. Used for getting stream info from test minibatch source.
    ///   * `labelInputName` - parameter of type `string`. Name of label input. Used for getting stream info from test minibatch source.
    ///   * `outputName` - parameter of type `string`. Name of final output node. Used for getting final output from model by name.
    ///   * `device` - parameter of type `DeviceDescriptor`. Device to run test.
    ///   * `maxCount` - parameter of type `int`. Number of samples to test.
    ///
    /// **Output Type**
    ///   * `float32` - Error rate for provided model.
    ///
    /// **Exceptions**
    ///
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

    
    /// **Description**  
    /// Save and reload model for given function.
    ///
    /// **Parameters**
    ///   * `func` - parameter of type `Function`. Function (in form of model) to save.
    ///   * `variables` - parameter of type `IList<Variable>`. Variables for this func. Input or output.
    ///   * `device` - parameter of type `DeviceDescriptor`. Device to generate new function on.
    ///   * `rank` - parameter of type `uint32`. Rank to use for model path.
    ///
    /// **Output Type**
    ///   * `Function`
    ///
    /// **Exceptions**
    ///
    static member SaveAndReloadModel
        (func: Function)
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
            
            func.Save(tempModelPath)
            let newFunc = Function.Load(tempModelPath, device)

            File.Delete(tempModelPath)

            let inputs = newFunc.Inputs
            for inputVariableInfo in inputVariableUids.ToList() do
                let newInputVariable = inputs.First(fun v -> v.Uid = inputVariableInfo.Key)
                inputVariableUids.[inputVariableInfo.Key] <- newInputVariable
            
            let outputs = newFunc.Outputs
            for outputVariableInfo in outputVariableNames.ToList() do
                let newOutputVariable = outputs.First(fun v -> v.Owner.Name = outputVariableInfo.Key)
                outputVariableNames.[outputVariableInfo.Key] <- newOutputVariable

            newFunc

    
    /// **Description**  
    /// Check whether minibatch data is sweep end, i.e. any of data within a minibatch marks sweep end.
    /// 
    /// **Parameters**
    ///   * `minibatchValues` - parameter of type `ICollection<MinibatchData>`. Minibatch data.
    ///
    /// **Output Type**
    ///   * `bool` - Whether minibatch is sweep end.
    ///
    /// **Exceptions**
    ///
    static member MiniBatchDataIsSweepEnd (minibatchValues: ICollection<MinibatchData>) =
        minibatchValues.Any(fun a -> a.sweepEnd)
       
    
    /// **Description**  
    /// Prints current training progress, obtained from trainer.
    ///
    /// **Parameters**
    ///   * `trainer` - parameter of type `Trainer`. Trainer to print training progress.
    ///   * `minibatchIdx` - parameter of type `int`. Current minibatch index.
    ///   * `outputFrequencyInMinibatches` - parameter of type `int`. How frequently printing progress should occur.
    ///
    /// **Output Type**
    ///   * `unit`
    ///
    /// **Exceptions**
    ///
    static member PrintTrainingProgress (trainer: Trainer) (minibatchIdx: int) (outputFrequencyInMinibatches: int) =
        if (minibatchIdx % outputFrequencyInMinibatches) = 0 && trainer.PreviousMinibatchSampleCount() <> 0u then
            let trainLossValue = (float)(trainer.PreviousMinibatchEvaluationAverage())
            let evaluationValue = (float)(trainer.PreviousMinibatchLossAverage())
            Console.WriteLine("Minibatch: {0} CrossEntropyLoss = {1}, EvaluationCriterion = {2}", minibatchIdx, trainLossValue, evaluationValue)

    
    /// **Description**  
    /// Prints output dimensions.
    ///
    /// **Parameters**
    ///   * `func` - parameter of type `Function`. Function to print dimension of output.
    ///   * `functionName` - parameter of type `string`. Name of function.
    ///
    /// **Output Type**
    ///   * `unit`
    ///
    /// **Exceptions**
    ///
    static member PrintOutputDims (func: Function) (functionName: string) =
        let shape = func.Output.Shape

        if shape.Rank = 3 then
            Console.WriteLine("{0} dim0: {1}, dim1: {2}, dim2: {3}", functionName, shape.[0], shape.[1], shape.[2])
        else
            Console.WriteLine("{0} dim0: {1}", functionName, shape.[0])
