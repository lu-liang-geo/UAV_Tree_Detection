This code provides a way to benchmark models on the NEON evaluation data
Examples are provided in the 'examples' folder

To run the benchmark on any model, follow these steps:
1. Create a new class that inherits from the ModelEvaluator class.
2. You MUST implement/override two functions in the child class
    a. load_model() : This returns an instance of your pretrained model
    b. predict_image() : Given an image in unnormalized RGB format, this returns a dicitionary containing bounding boxes and confidence scores (refer to function docstring for details)

To evaluate your model on the dataset, call the evaluate_model() function on an instance of your child class, this will return a dictionary containing bounding boxes, scores, and metrics (refer to docstring).

After evaluating the model, you can also plot images with the calculated bounding boxes using the plot_image_annotations() function
To evaluate the model on any desired input image, use the eval_and_plot_image_annotations() function