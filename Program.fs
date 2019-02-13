namespace CNTKExamples

open System
open CNTK

module Program =

    [<EntryPoint>]
    let main argv =
        printfn "Hello World from F#!"

        let device = DeviceDescriptor.CPUDevice
        printfn "========== running LogisticRegression.TrainAndEvaluate using %s ==========" (device.Type.ToString())
        LogisticRegression.TrainAndEvaluate device

        printfn "========== running MNISTClassifier.TrainAndEvaluate with MultiLayer Perceptron (MLP) classifier using %s ==========" (device.Type.ToString())
        MNISTClassifier.TrainAndEvaluate device false true

        printfn "========== running MNISTClassifier.TrainAndEvaluate with Convolutional Neural Netrowk (CNN) using %s ==========" (device.Type.ToString())
        MNISTClassifier.TrainAndEvaluate device true true

        0 // return an integer exit code
