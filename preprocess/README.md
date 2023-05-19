List of Sagemaker images can be find here:

https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-algo-docker-registry-paths.html

Note the region and account ID.


A docker login my be required. Use the following command or similar:

aws ecr get-login-password \
    --region us-east-2 \
| docker login \
    --username AWS \
    --password-stdin 257758044811.dkr.ecr.us-east-2.amazonaws.com

