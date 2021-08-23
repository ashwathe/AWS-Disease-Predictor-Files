FROM public.ecr.aws/lambda/python:3.7

COPY app.py requirements.txt degree_cancer_8.tflite ./

RUN python3.7 -m pip install --upgrade pip==20.1.1 

RUN python3.7 -m pip install -r requirements.txt -t .

# Command can be overwritten by providing a different command in the template directly.
CMD ["app.lambda_handler"]
