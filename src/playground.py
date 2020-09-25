# %%
import boto3
import awscli


def detect_labels(photo, bucket):
  '''processes an image from s3 bucket into Amazon Rekognition
   to determine objects in image + bounding box locations'''

  client=boto3.client('rekognition', region_name='us-east-1')

  response = client.detect_labels(Image={'S3Object':{'Bucket': bucket, 'Name':photo}},
        MaxLabels=5)

  print('Detected labels for ' + photo)
  print()
  for label in response['Labels']:
      print ("Label: " + label['Name'])
      print ("Confidence: " + str(label['Confidence']))
      print ("Instances:")
      for instance in label['Instances']:
          print ("  Bounding box")
          print ("    Top: " + str(instance['BoundingBox']['Top']))
          print ("    Left: " + str(instance['BoundingBox']['Left']))
          print ("    Width: " +  str(instance['BoundingBox']['Width']))
          print ("    Height: " +  str(instance['BoundingBox']['Height']))
          print ("  Confidence: " + str(instance['Confidence']))
          print()

      print ("Parents:")
      for parent in label['Parents']:
          print ("   " + parent['Name'])
      print ("----------")
      print ()
  return len(response['Labels'])


def main():
    photo='train/tudor/23.jpg'
    bucket='cbh-capstone3'
    label_count=detect_labels(photo, bucket)
    print("Labels detected: " + str(label_count))


if __name__ == "__main__":
    main()