# Spam Filter
*Automated spam tagging system for e-mails*

### Description
* SES pulls e-mails sent to an address and stores them in an S3 bucket
* Train an ML model to detect spam in text, and deploy it
* LF1 extracts body of the mail, and uses the Sagemaker endpoint to check for spam
* CloudFormation template represents all infrastructure and permissions as code

### Architecture
<img src="Images/architecture.png" width="600">

### Tech Stack
AWS (Lambda, S3, SES, Sagemaker, CloudFormation), Python 

### Contributors

* Vishal Prakash (vp2181@nyu.edu)
* Vedang Mondreti (vm2129@nyu.edu)