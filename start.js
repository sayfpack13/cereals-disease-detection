const { execFile } = require("child_process");


function runInferenceScript(modelPath, imagePath) {
    const args = ["start.py", "--model", modelPath, "--image", imagePath];
    execFile("python", args, (error, stdout, stderr) => {
        if (error) {
            console.error(`${stderr}`);
            return;
        }
        console.log(stdout);
    });
}


runInferenceScript("best_model.pth", "inference_images/image5.jpg");