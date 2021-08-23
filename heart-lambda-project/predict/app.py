import json
import tensorflow as tf
import logging
import warnings
import numpy as np

# Prevent multiprocessing warning in AWS logs
warnings.filterwarnings(action='ignore')

logger = logging.getLogger()
logger.setLevel(logging.INFO)

target_names = {
    "0": "Benign",
    "1": "Malignant"
}

# Load the TFLite model in TFLite Interpreter
interpreter = tf.lite.Interpreter('degree_heart.tflite')


def lambda_handler(event, context):
    # Load body and log content
    body = json.loads(event['body'])
    logger.info('Received request body: {}'.format(event['body']))

    # Check features are provided
    if 'features' not in body.keys():
        return {
                "statusCode": 400,
            "body": json.dumps(
                {
                    "message": "Invalid request. Missing parameters in body",
                }
            ),
        }

    features = np.array(body['features']).reshape(1, -1)
    logging.info('features: {}'.format(features))

    # Check feature dimensions
    if features.shape[-1] != 13:
        return {
            "statusCode": 400,
            "body": json.dumps(
                {
                    "message": "Invalid request. Received {} parameters, expected 13".format(features.shape[1]),
                }
            ),
        }

    # Calculate prediction
    try:
        interpreter.allocate_tensors()

        # Get input and output tensors.
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        interpreter.set_tensor(input_details[0]['index'], features.astype('float32'))
        interpreter.invoke()

        #prediction = model_pipeline.predict(features)
        output_data = interpreter.get_tensor(output_details[0]['index'])
        prediction_payload = {
            "class_label": str(output_data),
            "class_name": target_names.get(str(output_data))
        }
        return {
            "statusCode": 200,
            "body": json.dumps(
                {
                    "message": "Success",
                    "prediction": prediction_payload
                }
            ),
        }

    except Exception as e:
        logger.error('Unhandled error: {}'.format(e))
        return {
            "statusCode": 500,
            "body": json.dumps(
                {
                    "message": "Unhandled error",
                }
            ),
        }



