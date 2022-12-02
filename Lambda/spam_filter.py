import json
import boto3
import email
import os

import string
import sys
import numpy as np

from hashlib import md5

if sys.version_info < (3,):
    maketrans = string.maketrans
else:
    maketrans = str.maketrans
    
def vectorize_sequences(sequences, vocabulary_length):
    results = np.zeros((len(sequences), vocabulary_length))
    for i, sequence in enumerate(sequences):
       results[i, sequence] = 1. 
    return results

def one_hot_encode(messages, vocabulary_length):
    data = []
    for msg in messages:
        temp = one_hot(msg, vocabulary_length)
        data.append(temp)
    return data

def text_to_word_sequence(text, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=" "):
    if lower:
        text = text.lower()

    if sys.version_info < (3,):
        if isinstance(text, unicode):
            translate_map = dict((ord(c), unicode(split)) for c in filters)
            text = text.translate(translate_map)
        elif len(split) == 1:
            translate_map = maketrans(filters, split * len(filters))
            text = text.translate(translate_map)
        else:
            for c in filters:
                text = text.replace(c, split)
    else:
        translate_dict = dict((c, split) for c in filters)
        translate_map = maketrans(translate_dict)
        text = text.translate(translate_map)
    
    print('Cleaned Text: ', text)
    seq = text.split(split)
    return [i for i in seq if i]

def one_hot(text, n, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=' '):

    return hashing_trick(text, n, hash_function='md5', filters=filters, lower=lower, split=split)


def hashing_trick(text, n, hash_function=None, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=' '):
    if hash_function is None:
        hash_function = hash
    elif hash_function == 'md5':
        hash_function = lambda w: int(md5(w.encode()).hexdigest(), 16)

    seq = text_to_word_sequence(text,
                                filters=filters,
                                lower=lower,
                                split=split)
    return [int(hash_function(w) % (n - 1) + 1) for w in seq]

def lambda_handler(event, context):
    #print(event)
    
    #variables
    endpoint_name=os.environ['endpoint_name']
    bucket=event['Records'][0]['s3']['bucket']['name']
    key=event['Records'][0]['s3']['object']['key']
    #bucket='a3-email-bucket'#event['Records'][0]['s3']['bucket']['name']#
    #key='46jv88bvfcv4taer57ot0d88hpntfp0kmbl7ghg1'#event['Records'][0]['s3']['object']['key']#
    

    #fetch object
    session = boto3.Session()
    s3_session = session.client('s3')
    response = s3_session.get_object(Bucket=bucket, Key=key)
    
    #read email
    email_obj = email.message_from_bytes(response['Body'].read())
    from_email = email_obj.get('From')
    body = email_obj.get_payload()[0].get_payload()
    #print(body)
    #print(from_email)
    
    #fetch output from model endpoint
    endpoint_name = 'sms-spam-classifier-mxnet-2022-11-29-00-34-48-632'
    runtime = session.client('runtime.sagemaker')
    vocabulary_length = 9013
    input_mail = [body.strip()]
    print(input_mail)
    temp_1 = one_hot_encode(input_mail, vocabulary_length)
    input_mail = vectorize_sequences(temp_1, vocabulary_length)
    print(input_mail)
    data = json.dumps(input_mail.tolist())
    response = runtime.invoke_endpoint(EndpointName=endpoint_name, ContentType='application/json', Body=data)
    print(response)
    
    res = json.loads(response["Body"].read())
    
    if res['predicted_label'][0][0] == 0:
        label = 'Ok'
    else:
        label = 'Spam'
    score = round(res['predicted_probability'][0][0], 4)
    score = score*100


    message = "We received your email sent at " + str(email_obj.get('To')) + " with the subject " + str(email_obj.get('Subject')) + ".\nHere \
is a 240 character sample of the email body:\n\n" + body[:240] + "\nThe email was \
categorized as " + str(label) + " with a " + str(score) + "% confidence."

    print(message)
    
    #send email
    email_client = session.client('ses')
    response_email = email_client.send_email(
        Destination={'ToAddresses': [from_email]},
        Message={
            'Body': {
                'Text': {
                    'Charset': 'UTF-8',
                    'Data': message,
                },
            },
            'Subject': {
                'Charset': 'UTF-8',
                'Data': 'Spam analysis of your email',
            },
        },
        Source=str(email_obj.get('To')),
    )
    print(response_email)
    return {}
