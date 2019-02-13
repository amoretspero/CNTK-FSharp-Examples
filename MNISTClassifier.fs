namespace CNTKExamples

open System
open System.IO
open System.Collections.Generic
open System.Linq
open System.Net
open CNTK
open System.Text

type MNISTClassifier =
    
    /// **Description**  
    /// Number of output classes.  
    /// 
    static member numOutputClasses = 10

    
    /// **Description**  
    /// Loads images from MNIST database, then generates minibatch source for training and testing.
    ///
    /// **Parameters**  
    /// (None)
    ///
    /// **Output Type**
    ///   * `MinibatchSource * MinibatchSource` - First of tuple is minibatch source for training, second is for testing.
    ///
    /// **Exceptions**
    ///
    static member ImageLoader() =
        let downloadPath = Directory.GetCurrentDirectory()
        let dataPath = Directory.GetCurrentDirectory()

        let trainImagesUrl = "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz"
        let trainImagesFileName = "cntk-mnist-train-images"
        let trainLabelsUrl = "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz"
        let trainLabelsFileName = "cntk-mnist-train-labels"
        let testImagesUrl = "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz"
        let testImagesFileName = "cntk-mnist-test-images"
        let testLabelsUrl = "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"
        let testLabelsFileName = "cntk-mnist-test-labels"

        let trainDataFileName = "cntk-mnist-train-data.txt"
        let testDataFileName = "cntk-mnist-test-data.txt"

        let client = new WebClient()
        
        printfn "Now downloading and reading training image dataset..."
        if Path.Combine(downloadPath, trainImagesFileName) |> File.Exists then
            printfn "Training image dataset already exists. Skipping downloading..."
        else        
            client.DownloadFile(trainImagesUrl, Path.Combine(downloadPath, trainImagesFileName))
        let trainImagesBuffer = Helper.DecompressGZip (File.ReadAllBytes(Path.Combine(downloadPath, trainImagesFileName)))
        let trainImagesMagicNumber = BitConverter.ToInt32(trainImagesBuffer.[0..3].Reverse().ToArray(), 0)
        printfn "Magic number: %d" trainImagesMagicNumber
        let trainImagesImagesCount = BitConverter.ToInt32(trainImagesBuffer.[4..7].Reverse().ToArray(), 0)
        printfn "Images count: %d" trainImagesImagesCount
        let rowCount = BitConverter.ToInt32(trainImagesBuffer.[8..11].Reverse().ToArray(), 0)
        printfn "Row count: %d" rowCount
        let columnCount = BitConverter.ToInt32(trainImagesBuffer.[12..15].Reverse().ToArray(), 0)
        printfn "Column count: %d" columnCount
        printfn "%d" trainImagesBuffer.Length
        let trainImages = [|
            for i in 0..(trainImagesImagesCount - 1) ->
                trainImagesBuffer.[(16 + i * rowCount * columnCount)..(16 + (i + 1) * rowCount * columnCount - 1)].Reverse().ToArray()
        |]
        printfn "Finished downloading and reading training image dataset."

        printfn "Now downloading and reading training label dataset..."
        if Path.Combine(downloadPath, trainLabelsFileName) |> File.Exists then
            printfn "Training label dataset already exists. Skipping downloading..."
        else
            client.DownloadFile(trainLabelsUrl, Path.Combine(downloadPath, trainLabelsFileName))
        let trainLabelsBuffer = Helper.DecompressGZip (File.ReadAllBytes(Path.Combine(downloadPath, trainLabelsFileName)))
        let trainLabelsMagicNumber = BitConverter.ToInt32(trainLabelsBuffer.[0..3].Reverse().ToArray(), 0)
        printfn "Magic number: %d" trainLabelsMagicNumber
        let trainLabelsCount = BitConverter.ToInt32(trainLabelsBuffer.[4..7].Reverse().ToArray(), 0)
        printfn "Labels count: %d" trainLabelsCount
        let trainLabels = [|
            for i in 0..(trainLabelsCount - 1) ->
                BitConverter.ToInt32([| trainLabelsBuffer.[(8 + i)]; 0uy; 0uy; 0uy |], 0)
        |]
        printfn "Finished downloading and reading training label dataset."

        printfn "Now downloading and reading test image dataset..."
        if Path.Combine(downloadPath, testImagesFileName) |> File.Exists then
            printfn "Test image dataset already exists. Skipping downloading..."
        else
            client.DownloadFile(testImagesUrl, Path.Combine(downloadPath, testImagesFileName))
        let testImagesBuffer = Helper.DecompressGZip (File.ReadAllBytes(Path.Combine(downloadPath, testImagesFileName)))
        let testImagesMagicNumber = BitConverter.ToInt32(testImagesBuffer.[0..3].Reverse().ToArray(), 0)
        printfn "Magic number: %d" testImagesMagicNumber
        let testImagesImagesCount = BitConverter.ToInt32(testImagesBuffer.[4..7].Reverse().ToArray(), 0)
        printfn "Images count: %d" testImagesImagesCount
        let rowCount = BitConverter.ToInt32(testImagesBuffer.[8..11].Reverse().ToArray(), 0)
        printfn "Row count: %d" rowCount
        let columnCount = BitConverter.ToInt32(testImagesBuffer.[12..15].Reverse().ToArray(), 0)
        printfn "Column count: %d" columnCount
        let testImages = [|
            for i in 0..(testImagesImagesCount - 1) ->
                testImagesBuffer.[(16 + i * rowCount * columnCount)..(16 + (i + 1) * rowCount * columnCount - 1)].Reverse().ToArray()
        |]
        printfn "Finished downloading and reading test image dataset."

        printfn "Now downloading and reading test label dataset..."
        if Path.Combine(downloadPath, testLabelsFileName) |> File.Exists then
            printfn "Test label dataset already exists. Skipping downloading..."
        else
            client.DownloadFile(testLabelsUrl, Path.Combine(downloadPath, testLabelsFileName))
        let testLabelsBuffer = Helper.DecompressGZip (File.ReadAllBytes(Path.Combine(downloadPath, testLabelsFileName)))
        let testLabelsMagicNumber = BitConverter.ToInt32(testLabelsBuffer.[0..3].Reverse().ToArray(), 0)
        printfn "Magic number: %d" testLabelsMagicNumber
        let testLabelsCount = BitConverter.ToInt32(testLabelsBuffer.[4..7].Reverse().ToArray(), 0)
        printfn "Labels count: %d" testLabelsCount
        let testLabels = [|
            for i in 0..(testLabelsCount - 1) ->
                BitConverter.ToInt32([| testLabelsBuffer.[(8 + i)]; 0uy; 0uy; 0uy |], 0)
        |]
        printfn "Finished downloading and reading test label dataset."

        let makeOneHotArray (n: int) =
            Array.reduce (fun prv cur -> prv + " " + cur) [| for i in 0..(MNISTClassifier.numOutputClasses - 1) -> if i = n then "1" else "0" |]

        let makeFeaturesArray (byteArray: byte []) =
            Array.reduce (fun prv cur -> prv + " " + cur) (Array.map (fun b -> BitConverter.ToInt32([| b; 0uy; 0uy; 0uy |], 0).ToString()) byteArray)

        printfn "Now generating training data as CNTK Text Format..."
        let trainData =
            Array.map
                <| (fun td ->
                            StringBuilder()
                                .AppendFormat("|labels ")
                                .AppendFormat("{0} \t", (makeOneHotArray (fst td)))
                                .AppendFormat("|features ")
                                .AppendFormat("{0} ", (makeFeaturesArray (snd td)))
                                .ToString()
                        )
                <| (Array.zip
                    <| trainLabels
                    <| trainImages
                )
        File.WriteAllLines(Path.Combine(dataPath, trainDataFileName), trainData)
        printfn "Finished generating training data as CNTK Text Format."

        printfn "Now generating test data as CNTK Text Format..."
        let testData =
            Array.map
                <| (fun td -> StringBuilder()
                                .AppendFormat("|labels ")
                                .AppendFormat("{0} \t", (makeOneHotArray (fst td)))
                                .AppendFormat("|features ")
                                .AppendFormat("{0} ", (makeFeaturesArray (snd td)))
                                .ToString()
                        )
                <| (Array.zip
                    <| testLabels
                    <| testImages
                )
        File.WriteAllLines(Path.Combine(dataPath, testDataFileName), testData)
        printfn "Finished generating test data as CNTK Text Format."

        // For MNIST dataset, we have features stream and labels stream.
        let streamConfigurations = new StreamConfigurationVector([| 
                new StreamConfiguration("features", rowCount * columnCount);
                new StreamConfiguration("labels", MNISTClassifier.numOutputClasses)
            |])    

        // Generates CTF(CNTK-Text-Format) deserializer for training and testing data.
        let trainCTFDeserializer = CNTKLib.CTFDeserializer(Path.Combine(dataPath, trainDataFileName), streamConfigurations)
        let testCTFDeserializer = CNTKLib.CTFDeserializer(Path.Combine(dataPath, testDataFileName), streamConfigurations)

        // Generates minibatch source configuration for training and testing data.
        let trainMinibatchSourceConfig = new MinibatchSourceConfig([| trainCTFDeserializer |])
        let testMinibatchSourceConfig = new MinibatchSourceConfig([| testCTFDeserializer |])

        // Generates minibatch source for training and testing.
        let trainMinibatchSource = CNTKLib.CreateCompositeMinibatchSource(trainMinibatchSourceConfig)
        let testMinibatchSource = CNTKLib.CreateCompositeMinibatchSource(testMinibatchSourceConfig)

        printfn "Finished generating data."

        (trainMinibatchSource, testMinibatchSource)

    
    
    /// **Description**  
    /// Creates an MLP(Multi-Layer Perceptron) classifier.  
    /// Here, we use 1-hidden layer MLP.
    /// First layer(input to 1st hidden layer) is fully connected and uses sigmoid activation function.
    /// Second layer(1st hidden layer to output layer) is also fully connected and uses no activation function.
    ///
    /// **Parameters**
    ///   * `device` - parameter of type `DeviceDescriptor`. Device to create this MLP classifier.
    ///   * `numOutputClasses` - parameter of type `int`. Number of output classes.
    ///   * `hiddenLayerDim` - parameter of type `int`. Dimension of 1st hidden layer(the only hidden layer in this MLP classifier).
    ///   * `scaledInput` - parameter of type `Function`. Input scaled to fit in range of `[0, 1]`.
    ///   * `classifierName` - parameter of type `string`. Name of this classifier.
    ///
    /// **Output Type**
    ///   * `Function` - MLP classifier can be viewed as composite function of fundamental building blocks CNTK provides that we used.
    ///
    /// **Exceptions**
    ///
    static member CreateMLPClassifier (device: DeviceDescriptor) (numOutputClasses: int) (hiddenLayerDim: int) (scaledInput: Function) (classifierName: string) =
        let dense1 = TestHelper.Dense scaledInput hiddenLayerDim device Activation.Sigmoid ""
        TestHelper.Dense dense1 numOutputClasses device Activation.None classifierName   
        
    
    /// **Description**  
    /// Generates a layer consisting of single convolution layer with ReLU activation functionand single max-pooling layer.
    /// We use Glorot-Uniform-Initializer with scale of 0.26 and max-pooling layer will auto-pad input for pooling window.
    ///
    /// **Parameters**
    ///   * `features` - parameter of type `Variable`. Input to this convolution with max-pooling layer. (Input data to a layer is called `features` in NN.)
    ///   * `device` - parameter of type `DeviceDescriptor`. Device to create this layer on.
    ///   * `kernelWidth` - parameter of type `int`. Width of kernel for convolution.
    ///   * `kernelHeight` - parameter of type `int`. Height of kernel for convolution.
    ///   * `numInputChannels` - parameter of type `int`. Number of channels for input, i.e. `features`.
    ///   * `outFeatureMapCount` - parameter of type `int`. Number of output feature maps. (`feature map` means activated output of a filter in NN. This `feature maps` act as `features` for next layer, if exists.)
    ///   * `hStride` - parameter of type `int`. Horizontal stride for convolution kernel.
    ///   * `vStride` - parameter of type `int`. Vertical stride for convolution kernal.
    ///   * `poolingWindowWidth` - parameter of type `int`. Width of max-pooling window.
    ///   * `poolingWindowHeight` - parameter of type `int`. Height of max-pooling window.
    ///
    /// **Output Type**
    ///   * `Function`. Output will be a function that computes convolution and max-pooling to input.
    ///
    /// **Exceptions**
    ///
    static member ConvolutionWithMaxPooling
        (features: Variable)
        (device: DeviceDescriptor)
        (kernelWidth: int)
        (kernelHeight: int)
        (numInputChannels: int)
        (outFeatureMapCount: int)
        (hStride: int)
        (vStride: int)
        (poolingWindowWidth: int)
        (poolingWindowHeight: int) =
            let convWScale = 0.26
            let convParams = new Parameter(NDShape.CreateNDShape([| kernelWidth; kernelHeight; numInputChannels; outFeatureMapCount |]), DataType.Float, CNTKLib.GlorotUniformInitializer(convWScale, -1, 2), device)
            let convFunction = CNTKLib.ReLU(new Variable(CNTKLib.Convolution(convParams, features, NDShape.CreateNDShape([| 1; 1; numInputChannels |]))))
            let pooling = CNTKLib.Pooling(new Variable(convFunction), PoolingType.Max, NDShape.CreateNDShape([| poolingWindowWidth; poolingWindowHeight |]), NDShape.CreateNDShape([| hStride; vStride |]), new BoolVector([| true |]))
            pooling
    
    
    /// **Description**  
    /// Creates convolutional neural network.  
    /// This network will be consisted with 2x convolution layer with max-pooling.  
    /// Kernel will have size of (3, 3).  
    /// Input will be 1-channel 2D image, output of first convolution layer will have 4-channel data(Number of feature maps is 4.).  
    /// Output of second convolution layer will have 8-channel data(Number of feature maps is 8.).  
    /// For both convolution layer, max-pooling window size will be (3, 3), stride of kernel be (2, 2).  
    /// After 2x convolution layer, we will use a single dense layer with no activation function.  
    ///
    /// **Parameters**
    ///   * `features` - parameter of type `Variable`. Input to this convolutional neural network.
    ///   * `outDims` - parameter of type `int`. Number of output dimension. This corresponds to number of output classes.
    ///   * `device` - parameter of type `DeviceDescriptor`. Device to create this convolutional neural network.
    ///   * `classifierName` - parameter of type `string`. Name of this network.
    ///
    /// **Output Type**
    ///   * `Function` - Output will be a function that computes two convolutions with max-pooling and a single dense layer on input.
    ///
    /// **Exceptions**
    ///
    static member CreateConvolutionNeuralNetwork (features: Variable) (outDims: int) (device: DeviceDescriptor) (classifierName: string) =
        let kernelWidth1, kernelHeight1, numInputChannels1, outFeatureMapCount1 = (3, 3, 1, 4)
        let hStride1, vStride1 = (2, 2)
        let poolingWindowWidth1, poolingWindowHeight1 = (3, 3)

        let pooling1 = MNISTClassifier.ConvolutionWithMaxPooling
                        <| features
                        <| device
                        <| kernelWidth1
                        <| kernelHeight1
                        <| numInputChannels1
                        <| outFeatureMapCount1
                        <| hStride1
                        <| vStride1
                        <| poolingWindowWidth1
                        <| poolingWindowHeight1
        let kernelWidth2, kernelHeight2, numInputChannels2, outFeatureMapCount2 = (3, 3, outFeatureMapCount1, 8)
        let hStride2, vStride2 = (2, 2)
        let poolingWindowWidth2, poolingWindowHeight2 = (3, 3)

        let pooling2 = MNISTClassifier.ConvolutionWithMaxPooling
                        <| new Variable(pooling1)
                        <| device
                        <| kernelWidth2
                        <| kernelHeight2
                        <| numInputChannels2
                        <| outFeatureMapCount2
                        <| hStride2
                        <| vStride2
                        <| poolingWindowWidth2
                        <| poolingWindowHeight2

        let denseLayer = TestHelper.Dense pooling2 outDims device Activation.None classifierName
        denseLayer

    
    /// **Description**  
    /// Trains a model and evaluates that.
    ///
    /// **Parameters**
    ///   * `device` - parameter of type `DeviceDescriptor`. Device to create, train and evaluate a model.
    ///   * `useConvolution` - parameter of type `bool`. If `true`, Convolutional neural network will be used, otherwise, MLP will be used.
    ///   * `forceRetrain` - parameter of type `bool`. If `true`, forcefully re-train model.
    ///
    /// **Output Type**
    ///   * `unit`
    ///
    /// **Exceptions**
    ///
    static member TrainAndEvaluate (device: DeviceDescriptor) (useConvolution: bool) (forceRetrain: bool) =
        let featureStreamName = "features"
        let labelStreamName = "labels"
        let classifierName = "classifierOutput"
        let imageDimForConvolution = NDShape.CreateNDShape([| 28; 28; 1 |])
        let imageDimForMLP = NDShape.CreateNDShape([| (28 * 28 * 1) |])
        let imageDim = if useConvolution then imageDimForConvolution else imageDimForMLP
        let imageSize = 28 * 28
        let numClasses = 10
        let minibatchMaxCount = 1000

        let modelFileNameForConvolution = "MNISTConvolution.model"
        let modelFileNameForMLP = "MNISTMLP.model"
        let modelFileName = if useConvolution then modelFileNameForConvolution else modelFileNameForMLP

        let trainMinibatchSource, testMinibatchSource = MNISTClassifier.ImageLoader()

        if File.Exists(modelFileName) && not forceRetrain then
            TestHelper.ValidateModelWithMinibatchSource
                <| modelFileName
                <| testMinibatchSource
                <| featureStreamName
                <| labelStreamName
                <| classifierName
                <| device
                <| minibatchMaxCount
                |> ignore
        else
            let input = CNTKLib.InputVariable(imageDim, DataType.Float, featureStreamName)
            let classifierOutput = 
                if useConvolution then
                    let scaledInput = CNTKLib.ElementTimes(Constant.Scalar<float32>(1.0f / 784.0f, device), input) // Float32 has been used since .NET API for CNTK does not support other data types.
                    MNISTClassifier.CreateConvolutionNeuralNetwork (new Variable(scaledInput)) numClasses device classifierName
                else
                    let hiddenLayerDim = 200
                    let scaledInput = CNTKLib.ElementTimes(Constant.Scalar<float32>(1.0f / 784.0f, device), input) // Float32 has been used since .NET API for CNTK does not support other data types.
                    MNISTClassifier.CreateMLPClassifier device numClasses hiddenLayerDim scaledInput classifierName
            
            let labels = CNTKLib.InputVariable(NDShape.CreateNDShape([| numClasses |]), DataType.Float, labelStreamName) // DataType.Float has been used since .NET API for CNTK does not support other data types.
            let trainingLoss = CNTKLib.CrossEntropyWithSoftmax(new Variable(classifierOutput), labels, "lossFunction")
            let prediction = CNTKLib.ClassificationError(new Variable(classifierOutput), labels, "classificationError")

            let featureStreamInfo = trainMinibatchSource.StreamInfo(featureStreamName)
            let labelStreamInfo = trainMinibatchSource.StreamInfo(labelStreamName)

            let learningRatePerSample = new CNTK.TrainingParameterScheduleDouble(0.003125, 1u)

            let parameterLearners = new List<Learner>()
            parameterLearners.Add(Learner.SGDLearner(classifierOutput.Parameters(), learningRatePerSample))
            let trainer = Trainer.CreateTrainer(classifierOutput, trainingLoss, prediction, parameterLearners)

            let minibatchSize = 64u
            let outputFrequencyInMinibatches = 20
            let mutable minibatchIdx = 0
            let mutable epoch = 5
            while epoch > 0 do
                let minibatchData = trainMinibatchSource .GetNextMinibatch(minibatchSize, device)

                let arguments = new Dictionary<Variable, MinibatchData>()
                arguments.Add(input, minibatchData.[featureStreamInfo])
                arguments.Add(labels, minibatchData.[labelStreamInfo])

                trainer.TrainMinibatch(arguments, device) |> ignore
                TestHelper.PrintTrainingProgress trainer minibatchIdx outputFrequencyInMinibatches

                if (TestHelper.MiniBatchDataIsSweepEnd minibatchData.Values) then epoch <- epoch - 1

                minibatchIdx <- minibatchIdx + 1

            classifierOutput.Save(modelFileName)

            TestHelper.ValidateModelWithMinibatchSource
                <| modelFileName
                <| testMinibatchSource
                <| featureStreamName
                <| labelStreamName
                <| classifierName
                <| device
                <| minibatchMaxCount
                |> ignore
