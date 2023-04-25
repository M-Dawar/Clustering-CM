import pymongo
import json
from bson.json_util import dumps
from pymongo import MongoClient
from datetime import datetime
import copy
import JsonToCsvConversion as convertToCsv

client = MongoClient("localhost", 27017)

db = client["coursemapper_v2"]
collection = db["activities"]
statements = []
activities = collection.find({})
# List of dictionaries (studentActivitiesDict JSON) for each student engagement record. Length = No of unique students
listOfStudentActivityDict = []
studentActivitiesDict = {
    'stdProfile': {
        'stdId': 0000,
        'stdUsername': "",
        'totalSessionTime': 0000,
        'avgSessionTime': 0000,
        'maxSessionTime': 0000,
        'minSessionTime': 0000,
        'totalEnrollments': 0000
    },
    'activitiesProfile': {

        'totalActivities': 0000,
        'annotations': {
            'totalAnnotations': 0000,
            'pdf': {
                'totalPdfAnnotations': 0000,
                'annotationCountTypewise': {
                    'noteTypeAnnotationsOnPdf': 0000,
                    'questionTypeAnnotationsOnPdf': 0000,
                    'questionTypeAnnotationsOnPdf': 0000,
                    'comment': {
                        'noteTypeCommentsOnPdf': 0000,
                        'questionTypeCommentsOnPdf': 0000,
                        'externalResourceTypeCommentsOnPdf': 0000,
                    }
                },
                'annotationTools': {
                    'highlightToolOnPdf': 0000,
                    'pinpointToolOnPdf': 0000,
                    'drawToolOnPdf': 0000,
                }
            },

            'video': {
                'totalVideoAnnotations': 0000,
                'annotationCountTypewise': {
                    'noteTypeAnnotationsOnVid': 0000,
                    'questionTypeAnnotationsOnVid': 0000,
                    'externalResourceTypeAnnotationsOnVid': 0000,
                    'comment': {
                        'noteTypeCommentsOnVid': 0000,
                        'questionTypeCommentsOnVid': 0000,
                        'externalResourceTypeCommentsOnVid': 0000,
                    }
                },
                'annotationTools': {
                    'highlightToolOnVid': 0000,
                    'pinpointToolOnVid': 0000,
                    'drawToolOnVid': 0000,
                }
            }
        },
        'likes': {
            'likesOnAnnotations': {
                'totalLikesOnAnnotatios': 0000,
                'likesOnNoteTypeAnnotations': 0000,
                'likesOnQuestionTypeAnnotations': 0000,
                'likesOnExternalResourceTypeAnnotations': 0000,
            },
            'likesOnComments': {
                'likesOnNoteTypeComments': 0000,
                'likesOnquestionTypeComments': 0000,
                'likesOnexternalResourceTypeComments': 0000,
            },
            'likesOnRepliesOfAnnotations': 0000
        },
        'dislikes': {
            'dislikesOnAnnotations': {
                'totalDislikesOnAnnotations': 0000,
                'dislikesOnNoteTypeAnnotations': 0000,
                'dislikesOnQuestionTypeAnnotations': 0000,
                'dislikesOnExternalResourceTypeAnnotations': 0000,
            },
            'dislikesOnComments': {
                'dislikesOnNoteTypeComments': 0000,
                'dislikesOnquestionTypeComments': 0000,
                'dislikesOnexternalResourceTypeComments': 0000,
            },
            'dislikesOnRepliesOfAnnotations': 0000
        },
        'access': {
            'totalAccesses': 0000,
            'courseAccesses': 0000,
            'topicAccesses': 0000,
            'channelAccesses': 0000,
            'materialAccesses': {
                'pdfAccess': 0000,
                'videoAccess': 0000
            }
        },
        'materialProfile': {
            'video': {
                'videosStarted': 0000,
                'videosCompleted': 0000,
                'videosPlayed': 0000,
                'videosPauses': 0000,
                'timeSpentOnVideos': 00.00

            },
            'pdf': {
                'pdfStarted': 0000,
                'pdfCompleted': 0000,
                'slidesViewed':0000,
            }
        }

    }
}

def writeToJsonFile(data, filename='studentsActivities.json'):
    with open(filename,"w") as f:
        json.dump(data,f,indent=2)

for document in activities:  # Get the xApi statements containing SOV from activities collection
    # print(document)
    statements.append(document['statement'])

aggr_activities = collection.aggregate([
    {
        "$group": {
            '_id': '$statement.actor.account.name',
            'totalActivities': {'$sum': 1},
            'activities': {
                '$push': {
                    'verb': '$statement.verb',
                    'object': '$statement.object',
                    'result': '$statement.result',
                    'context': '$statement.context',
                    'timestamp': '$statement.timestamp',
                    'totalActivities': '$sum'
                }
            }
        }
    },  # -1 to sort by latest to oldest
    {
        "$sort": {'statement.timestamp': 1}
    }
])

# for i in aggr_activities:
#   print(i)

# Counts the number of lists created: Should be equal to number of students (Objects) returned from aggregation
listCount = 0
# List of lists (containing all activities for each students)
studentActivitiesListAggregated = []

video = {
    'videoId':00,
    'playedDurations':00,
    'pausedDurations':00
}
allVideos=[video]
startedVideosIDList = []

for i in aggr_activities:  # To clean the aggregated object and get only id, verb, object and timestamp and push into their each separate list
    studentActivitiesListAggregated.append([])
    listOfStudentActivityDict.append(copy.deepcopy(studentActivitiesDict))
    #print(i)
    # print(len(i))
    # print(i['_id'])
    for j in range(len(i['activities'])):
        if('annotated material' in i['activities'][j]['verb']['display']['en-US']+" " + i['activities'][j]['object']['definition']['type'].rsplit('/', 1)[-1]):
            #  Annotation material activity: [username annotated material material_type annotation_type annotation_tool timestamp]
            studentActivitiesListAggregated[listCount].append(i['_id']+" " + i['activities'][j]['verb']['display']['en-US']+" " + i['activities'][j]['object']['definition']['type'].rsplit('/', 1)[-1] + " "+i['activities'][j]['object']['definition']['extensions']['http://www.CourseMapper.de/extensions/material']
                                                              ['type'] + " " + i['activities'][j]['result']['extensions']['http://www.CourseMapper.de/extensions/annotation']['type']+" "+i['activities'][j]['result']['extensions']['http://www.CourseMapper.de/extensions/annotation']['tool']['type'] + " " + str(i['activities'][j]['timestamp']))
            #  Liked annotation activity: [username liked annotation material_type annotation'_type annotation_tool timestamp]
        elif('liked annotation' in i['activities'][j]['verb']['display']['en-US']+" " + i['activities'][j]['object']['definition']['type'].rsplit('/', 1)[-1]):
            studentActivitiesListAggregated[listCount].append(i['_id']+" " + i['activities'][j]['verb']['display']['en-US']+" " + i['activities'][j]['object']['definition']['type'].rsplit('/', 1)[-1] + " " + i['activities'][j]['object']['definition']['extensions']
                                                              ['http://www.CourseMapper.de/extensions/annotation']['type']+" "+i['activities'][j]['object']['definition']['extensions']['http://www.CourseMapper.de/extensions/annotation']['tool']['type'] + " " + str(i['activities'][j]['timestamp']))
            #   Accessed material activity: [username accessed material material_type timestamp]
        elif('accessed material' in i['activities'][j]['verb']['display']['en-US']+" " + i['activities'][j]['object']['definition']['type'].rsplit('/', 1)[-1]):
            studentActivitiesListAggregated[listCount].append(i['_id']+" " + i['activities'][j]['verb']['display']['en-US']+" " + i['activities'][j]['object']['definition']['type'].rsplit('/', 1)[-1] + " " + i['activities'][j]['object']['definition']['extensions']
                                                              ['http://www.CourseMapper.de/extensions/material']['type'] + " " + str(i['activities'][j]['timestamp']))
            #   Completed video/pdf activity [username completed material_type timestamp]
        elif('completed' in i['activities'][j]['verb']['display']['en-US']+" " + i['activities'][j]['object']['definition']['type'].rsplit('/', 1)[-1]):
            studentActivitiesListAggregated[listCount].append(i['_id']+" " + i['activities'][j]['verb']['display']['en-US']+" " + i['activities'][j]['object']['definition']['type'].rsplit('/', 1)[-1] + " " + i['activities'][j]['object']['definition']['extensions']
                                                              ['http://www.CourseMapper.de/extensions/material']['type'] + " " + str(i['activities'][j]['timestamp']))
            #   Played video activity [username played/paused video video_id duration(seconds) timestamp]
            #   If start duration is 0 seconds, means the video is just started
        elif('played' in i['activities'][j]['verb']['display']['en-US']+" " + i['activities'][j]['object']['definition']['type'].rsplit('/', 1)[-1]):
            studentActivitiesListAggregated[listCount].append(i['_id']+" " + i['activities'][j]['verb']['display']['en-US']+" " + i['activities'][j]['object']['definition']['type'].rsplit('/', 1)[-1] + " " + i['activities'][j]['object']['definition']['extensions']['http://www.CourseMapper.de/extensions/material']['type'] +" " +
                                                            i['activities'][j]['object']['definition']['extensions']['http://www.CourseMapper.de/extensions/material']['timestamp']+" " + str(i['activities'][j]['timestamp']))
            
            #   Paused video activity [username played/paused video video_id duration(seconds) timestamp] 
        elif('paused' in i['activities'][j]['verb']['display']['en-US']+" " + i['activities'][j]['object']['definition']['type'].rsplit('/', 1)[-1]):
            studentActivitiesListAggregated[listCount].append(i['_id']+" " + i['activities'][j]['verb']['display']['en-US']+" " + i['activities'][j]['object']['definition']['type'].rsplit('/', 1)[-1] + " " + i['activities'][j]['object']['definition']['extensions']['http://www.CourseMapper.de/extensions/material']['type'] +" " +
                                                            i['activities'][j]['object']['definition']['extensions']['http://www.CourseMapper.de/extensions/material']['timestamp']+" " + str(i['activities'][j]['timestamp']))
            #   Viewed slide activity [username viewed slide slide_number timestamp]
            #   If the slide_number is 0, means the material is started???????
        elif('viewed slide' in i['activities'][j]['verb']['display']['en-US']+" " + i['activities'][j]['object']['definition']['type'].rsplit('/', 1)[-1]):
            studentActivitiesListAggregated[listCount].append(i['_id']+" " + i['activities'][j]['verb']['display']['en-US']+" " + i['activities'][j]['object']['definition']['type'].rsplit('/', 1)[-1] + " " + i['activities'][j]['object']['definition']['extensions']
                                                              ['http://www.CourseMapper.de/extensions/material']['pageNr'] + " " + str(i['activities'][j]['timestamp']))
        else:  # On any other activity [username verb object timestamp]
            studentActivitiesListAggregated[listCount].append(i['_id']+" " + i['activities'][j]['verb']['display']['en-US']+" " +
                                                              i['activities'][j]['object']['definition']['type'].rsplit('/', 1)[-1] + " " + str(i['activities'][j]['timestamp']))
        #print(i['_id']+" " + i['activities'][j]['verb']['display']['en-US'], i['activities'][j]['object']['definition']['type'].rsplit('/', 1)[-1], i['activities'][j]['timestamp'])
        # print(j)
    #print(studentActivitiesListAggregated[listCount])
    listCount = listCount+1
 #   print(i)

#print(studentActivitiesListAggregated)
# Converting timestamp string to python identifiable format, to get the time in seconds or minutes
""" print(datetime.now())
login='2023-03-02 20:16:07.552000'
logout='2023-02-09 12:44:56.299000'
loginT=datetime.strptime(login, "%Y-%m-%d %H:%M:%S.%f")
logoutT=datetime.strptime(logout, "%Y-%m-%d %H:%M:%S.%f")
diff=logoutT-loginT
diff.seconds/60 """


""" print(stdActivitiesJSON['activitiesProfile']['totalActivities'])
jsonString = json.dumps(stdActivitiesJSON, indent=2)
#print(jsonString[0]) """
# print(stdActivitiesDict['stdProfile']['totalEnrollments'])


# print(stdActivitiesDict['stdProfile']['totalEnrollments'])
""" 
for i in range(listCount):  #Create list of dictionaries as many as number of results from aggregation (listindex)
    listOfStudentActivityDict.append(studentActivitiesDict) """

# print(listOfStudentActivityDict)

# print(type(listOfStudentActivityDict))
# print(type(listOfStudentActivityDict[0]))

'''
for ActList in stdActList:
    for activity in ActList:
        if 'enrolled course' in activity: #No. of enrollments
           studentActivitiesDict['stdProfile']['totalEnrollments']=studentActivitiesDict['stdProfile']['totalEnrollments']+1
        if 'completed pdf' in activity: #pdf completed
            studentActivitiesDict['activitiesProfile']['materialCompletion']['pdf']=studentActivitiesDict['activitiesProfile']['materialCompletion']['pdf']+1
'''

# print(range(len(listOfStudentActivityDict)))
# print(range(len(studentActivitiesListAggregated)))


# for i in range(len(listOfStudentActivityDict)):
#    print(listOfStudentActivityDict[i])
# for ActivityList in studentActivitiesListAggregated:
#    print(ActivityList)
#   for activity in ActivityList:
#      if 'enrolled course' in activity: #No. of enrollments
#         listOfStudentActivityDict[i]['stdProfile']['totalEnrollments']=listOfStudentActivityDict[i]['stdProfile']['totalEnrollments']+1


# for i in listOfStudentActivityDict:
#   print(i)




for listIndex in range(len(studentActivitiesListAggregated)):
    listOfStudentActivityDict[listIndex]['stdProfile']['stdUsername'] = studentActivitiesListAggregated[listIndex][0].split(' ')[
        0]
    listOfStudentActivityDict[listIndex]['stdProfile']['stdId'] = listIndex+1
    for itemIndex in range(len(studentActivitiesListAggregated[listIndex])):

        # Total activities
        listOfStudentActivityDict[listIndex]['activitiesProfile']['totalActivities'] = len(
            studentActivitiesListAggregated[listIndex])

        # Count Total Enrollments
        if 'enrolled course' in studentActivitiesListAggregated[listIndex][itemIndex]:
            listOfStudentActivityDict[listIndex]['stdProfile'][
                'totalEnrollments'] = listOfStudentActivityDict[listIndex]['stdProfile']['totalEnrollments']+1

        # Max session time
        # Min session time
        # Avg session time

        # Annotations

        # Total Annotations
        elif 'annotated material' in studentActivitiesListAggregated[listIndex][itemIndex]:
            listOfStudentActivityDict[listIndex]['activitiesProfile']['annotations'][
                'totalAnnotations'] = listOfStudentActivityDict[listIndex]['activitiesProfile']['annotations']['totalAnnotations']+1

            # Total PDF Annotations
            if 'annotated material pdf' in studentActivitiesListAggregated[listIndex][itemIndex]:
                listOfStudentActivityDict[listIndex]['activitiesProfile']['annotations']['pdf'][
                    'totalPdfAnnotations'] = listOfStudentActivityDict[listIndex]['activitiesProfile']['annotations']['pdf']['totalPdfAnnotations']+1

                # PDF Annotation Tools
                if 'highlight' in studentActivitiesListAggregated[listIndex][itemIndex]:
                    listOfStudentActivityDict[listIndex]['activitiesProfile']['annotations']['pdf']['annotationTools'][
                        'highlight'] = listOfStudentActivityDict[listIndex]['activitiesProfile']['annotations']['pdf']['annotationTools']['highlight']+1
                if 'pinpoint' in studentActivitiesListAggregated[listIndex][itemIndex]:
                    listOfStudentActivityDict[listIndex]['activitiesProfile']['annotations']['pdf']['annotationTools'][
                        'pinpoint'] = listOfStudentActivityDict[listIndex]['activitiesProfile']['annotations']['pdf']['annotationTools']['pinpoint']+1
                if 'draw' in studentActivitiesListAggregated[listIndex][itemIndex]:
                    listOfStudentActivityDict[listIndex]['activitiesProfile']['annotations']['pdf']['annotationTools'][
                        'draw'] = listOfStudentActivityDict[listIndex]['activitiesProfile']['annotations']['pdf']['annotationTools']['draw']+1

                # Annotation Types
                if 'annotated material pdf Question' in studentActivitiesListAggregated[listIndex][itemIndex]:
                    listOfStudentActivityDict[listIndex]['activitiesProfile']['annotations']['pdf']['annotationCountTypewise'][
                        'question'] = listOfStudentActivityDict[listIndex]['activitiesProfile']['annotations']['pdf']['annotationCountTypewise']['question']+1
                if 'annotated material pdf Note' in studentActivitiesListAggregated[listIndex][itemIndex]:
                    listOfStudentActivityDict[listIndex]['activitiesProfile']['annotations']['pdf']['annotationCountTypewise'][
                        'note'] = listOfStudentActivityDict[listIndex]['activitiesProfile']['annotations']['pdf']['annotationCountTypewise']['note']+1
                if 'annotated material pdf External Resource' in studentActivitiesListAggregated[listIndex][itemIndex]:
                    listOfStudentActivityDict[listIndex]['activitiesProfile']['annotations']['pdf']['annotationCountTypewise'][
                        'externalResource'] = listOfStudentActivityDict[listIndex]['activitiesProfile']['annotations']['pdf']['annotationCountTypewise']['externalResource']+1

                # Comments and types (Annotations without using tool)
                if 'annotated material pdf External Resource annotation' in studentActivitiesListAggregated[listIndex][itemIndex]:
                    listOfStudentActivityDict[listIndex]['activitiesProfile']['annotations']['pdf']['annotationCountTypewise']['comment'][
                        'externalResource'] = listOfStudentActivityDict[listIndex]['activitiesProfile']['annotations']['pdf']['annotationCountTypewise']['comment']['externalResource']+1
                if 'annotated material pdf Question annotation' in studentActivitiesListAggregated[listIndex][itemIndex]:
                    listOfStudentActivityDict[listIndex]['activitiesProfile']['annotations']['pdf']['annotationCountTypewise']['comment'][
                        'question'] = listOfStudentActivityDict[listIndex]['activitiesProfile']['annotations']['pdf']['annotationCountTypewise']['comment']['question']+1
                if 'annotated material pdf Note annotation' in studentActivitiesListAggregated[listIndex][itemIndex]:
                    listOfStudentActivityDict[listIndex]['activitiesProfile']['annotations']['pdf']['annotationCountTypewise']['comment'][
                        'note'] = listOfStudentActivityDict[listIndex]['activitiesProfile']['annotations']['pdf']['annotationCountTypewise']['comment']['note']+1

             #####################################    PDF     #############################################################

             # Video Annotations
             # Total Video Annotations
            elif 'annotated material video' in studentActivitiesListAggregated[listIndex][itemIndex]:
                listOfStudentActivityDict[listIndex]['activitiesProfile']['annotations']['video'][
                    'totalVideoAnnotations'] = listOfStudentActivityDict[listIndex]['activitiesProfile']['annotations']['pdf']['totalVideoAnnotations']+1

                """  # Video Annotation Tools Usage
                if 'highlight' in studentActivitiesListAggregated[listIndex][itemIndex]:
                    listOfStudentActivityDict[listIndex]['activitiesProfile']['annotations']['pdf']['annotationTools'][
                        'highlight'] = listOfStudentActivityDict[listIndex]['activitiesProfile']['annotations']['pdf']['annotationTools']['highlight']+1
                if 'pinpoint' in studentActivitiesListAggregated[listIndex][itemIndex]:
                    listOfStudentActivityDict[listIndex]['activitiesProfile']['annotations']['pdf']['annotationTools'][
                        'pinpoint'] = listOfStudentActivityDict[listIndex]['activitiesProfile']['annotations']['pdf']['annotationTools']['pinpoint']+1
                if 'draw' in studentActivitiesListAggregated[listIndex][itemIndex]:
                    listOfStudentActivityDict[listIndex]['activitiesProfile']['annotations']['pdf']['annotationTools'][
                        'draw'] = listOfStudentActivityDict[listIndex]['activitiesProfile']['annotations']['pdf']['annotationTools']['draw']+1
                """
                # Annotation Types
                if 'annotated material video Question' in studentActivitiesListAggregated[listIndex][itemIndex]:
                    listOfStudentActivityDict[listIndex]['activitiesProfile']['annotations']['video']['annotationCountTypewise'][
                        'question'] = listOfStudentActivityDict[listIndex]['activitiesProfile']['annotations']['video']['annotationCountTypewise']['question']+1
                if 'annotated material video Note' in studentActivitiesListAggregated[listIndex][itemIndex]:
                    listOfStudentActivityDict[listIndex]['activitiesProfile']['annotations']['video']['annotationCountTypewise'][
                        'note'] = listOfStudentActivityDict[listIndex]['activitiesProfile']['annotations']['video']['annotationCountTypewise']['note']+1
                if 'annotated material video External Resource' in studentActivitiesListAggregated[listIndex][itemIndex]:
                    listOfStudentActivityDict[listIndex]['activitiesProfile']['annotations']['video']['annotationCountTypewise'][
                        'externalResource'] = listOfStudentActivityDict[listIndex]['activitiesProfile']['annotations']['video']['annotationCountTypewise']['externalResource']+1

                 # Comments and types (Annotations without using tool)
                if 'annotated material video External Resource annotation' in studentActivitiesListAggregated[listIndex][itemIndex]:
                    listOfStudentActivityDict[listIndex]['activitiesProfile']['annotations']['video']['annotationCountTypewise']['comment'][
                        'externalResource'] = listOfStudentActivityDict[listIndex]['activitiesProfile']['annotations']['video']['annotationCountTypewise']['comment']['externalResource']+1

                if 'annotated material video Question annotation' in studentActivitiesListAggregated[listIndex][itemIndex]:
                    listOfStudentActivityDict[listIndex]['activitiesProfile']['annotations']['video']['annotationCountTypewise']['comment'][
                        'question'] = listOfStudentActivityDict[listIndex]['activitiesProfile']['annotations']['video']['annotationCountTypewise']['comment']['question']+1

                if 'annotated material video Note annotation' in studentActivitiesListAggregated[listIndex][itemIndex]:
                    listOfStudentActivityDict[listIndex]['activitiesProfile']['annotations']['video']['annotationCountTypewise']['comment'][
                        'note'] = listOfStudentActivityDict[listIndex]['activitiesProfile']['annotations']['video']['annotationCountTypewise']['comment']['note']+1

        
        # Count Total likes on annotations, comments and replies
        elif 'liked' in studentActivitiesListAggregated[listIndex][itemIndex]:
            splittedThisActivityStringArray=  studentActivitiesListAggregated[listIndex][itemIndex].split(' ')
            if  splittedThisActivityStringArray[1] == 'liked':
                if splittedThisActivityStringArray[2]=='annotation':
                    listOfStudentActivityDict[listIndex]['activitiesProfile']['likes']['likesOnAnnotations']['totalLikes']=listOfStudentActivityDict[listIndex]['activitiesProfile']['likes']['likesOnAnnotations']['totalLikes']+1
                    if splittedThisActivityStringArray[3] == 'Note':
                        listOfStudentActivityDict[listIndex]['activitiesProfile']['likes']['likesOnAnnotations']['note']= listOfStudentActivityDict[listIndex]['activitiesProfile']['likes']['likesOnAnnotations']['note']+1
                        if splittedThisActivityStringArray[4] == 'annotation':
                            listOfStudentActivityDict[listIndex]['activitiesProfile']['likes']['likesOnComments']['note']=listOfStudentActivityDict[listIndex]['activitiesProfile']['likes']['likesOnComments']['note']+1
                    elif splittedThisActivityStringArray[3] == 'Question':
                        listOfStudentActivityDict[listIndex]['activitiesProfile']['likes']['likesOnAnnotations']['question']= listOfStudentActivityDict[listIndex]['activitiesProfile']['likes']['likesOnAnnotations']['question']+1
                        if splittedThisActivityStringArray[4] == 'annotation':
                            listOfStudentActivityDict[listIndex]['activitiesProfile']['likes']['likesOnComments']['question']=listOfStudentActivityDict[listIndex]['activitiesProfile']['likes']['likesOnComments']['question']+1
                    elif splittedThisActivityStringArray[3] == 'External Resource':
                        listOfStudentActivityDict[listIndex]['activitiesProfile']['likes']['likesOnAnnotations']['externalResource']= listOfStudentActivityDict[listIndex]['activitiesProfile']['likes']['likesOnAnnotations']['externalResource']+1
                        if splittedThisActivityStringArray[4] == 'annotation':
                            listOfStudentActivityDict[listIndex]['activitiesProfile']['likes']['likesOnComments']['externalResource']=listOfStudentActivityDict[listIndex]['activitiesProfile']['likes']['likesOnComments']['externalResource']+1
                elif splittedThisActivityStringArray[2]=='reply':
                    listOfStudentActivityDict[listIndex]['activitiesProfile']['likes']['likesOnReplies']=listOfStudentActivityDict[listIndex]['activitiesProfile']['likes']['likesOnReplies']+1
        
        
        # Count the unlikes on annotations, comments, replies to subtract from the likes    
        elif 'unliked' in studentActivitiesListAggregated[listIndex][itemIndex]:
            splittedThisActivityStringArray=  studentActivitiesListAggregated[listIndex][itemIndex].split(' ')
            if  splittedThisActivityStringArray[1] == 'unliked':
                if splittedThisActivityStringArray[2]=='annotation':
                    listOfStudentActivityDict[listIndex]['activitiesProfile']['likes']['likesOnAnnotations']['totalLikes']=listOfStudentActivityDict[listIndex]['activitiesProfile']['likes']['likesOnAnnotations']['totalLikes']-1
                    if splittedThisActivityStringArray[3] == 'Note':
                        listOfStudentActivityDict[listIndex]['activitiesProfile']['likes']['likesOnAnnotations']['note']= listOfStudentActivityDict[listIndex]['activitiesProfile']['likes']['likesOnAnnotations']['note']-1
                        if splittedThisActivityStringArray[4] == 'annotation':
                            listOfStudentActivityDict[listIndex]['activitiesProfile']['likes']['likesOnComments']['note']=listOfStudentActivityDict[listIndex]['activitiesProfile']['likes']['likesOnComments']['note']-1
                    elif splittedThisActivityStringArray[3] == 'Question':
                        listOfStudentActivityDict[listIndex]['activitiesProfile']['likes']['likesOnAnnotations']['question']= listOfStudentActivityDict[listIndex]['activitiesProfile']['likes']['likesOnAnnotations']['question']-1
                        if splittedThisActivityStringArray[4] == 'annotation':
                            listOfStudentActivityDict[listIndex]['activitiesProfile']['likes']['likesOnComments']['question']=listOfStudentActivityDict[listIndex]['activitiesProfile']['likes']['likesOnComments']['question']-1
                    elif splittedThisActivityStringArray[3] == 'External Resource':
                        listOfStudentActivityDict[listIndex]['activitiesProfile']['likes']['likesOnAnnotations']['externalResource']= listOfStudentActivityDict[listIndex]['activitiesProfile']['likes']['likesOnAnnotations']['externalResource']-1
                        if splittedThisActivityStringArray[4] == 'annotation':
                            listOfStudentActivityDict[listIndex]['activitiesProfile']['likes']['likesOnComments']['externalResource']=listOfStudentActivityDict[listIndex]['activitiesProfile']['likes']['likesOnComments']['externalResource']-1
                elif splittedThisActivityStringArray[2]=='reply':
                    listOfStudentActivityDict[listIndex]['activitiesProfile']['likes']['likesOnReplies']=listOfStudentActivityDict[listIndex]['activitiesProfile']['likes']['likesOnReplies']-1

        # Count dislikes on annotations, comments and replies
        elif 'disliked' in studentActivitiesListAggregated[listIndex][itemIndex]:
            splittedThisActivityStringArray=  studentActivitiesListAggregated[listIndex][itemIndex].split(' ')
            if  splittedThisActivityStringArray[1] == 'disliked':
                if splittedThisActivityStringArray[2]=='annotation':
                    listOfStudentActivityDict[listIndex]['activitiesProfile']['dislikes']['dislikesOnAnnotations']['totalLikes']=listOfStudentActivityDict[listIndex]['activitiesProfile']['dislikes']['dislikesOnAnnotations']['totalDisikes']+1
                    if splittedThisActivityStringArray[3] == 'Note':
                        listOfStudentActivityDict[listIndex]['activitiesProfile']['dislikes']['dislikesOnAnnotations']['note']= listOfStudentActivityDict[listIndex]['activitiesProfile']['dislikes']['dislikesOnAnnotations']['note']+1
                        if splittedThisActivityStringArray[4] == 'annotation':
                            listOfStudentActivityDict[listIndex]['activitiesProfile']['dislikes']['dislikesOnAnnotations']['note']=listOfStudentActivityDict[listIndex]['activitiesProfile']['dislikes']['dislikesOnAnnotations']['note']+1
                    elif splittedThisActivityStringArray[3] == 'Question':
                        listOfStudentActivityDict[listIndex]['activitiesProfile']['dislikes']['dislikesOnAnnotations']['question']= listOfStudentActivityDict[listIndex]['activitiesProfile']['dislikes']['dislikesOnAnnotations']['question']+1
                        if splittedThisActivityStringArray[4] == 'annotation':
                            listOfStudentActivityDict[listIndex]['activitiesProfile']['dislikes']['dislikesOnAnnotations']['question']=listOfStudentActivityDict[listIndex]['activitiesProfile']['dislikes']['dislikesOnAnnotations']['question']+1
                    elif splittedThisActivityStringArray[3] == 'External Resource':
                        listOfStudentActivityDict[listIndex]['activitiesProfile']['dislikes']['dislikesOnAnnotations']['externalResource']= listOfStudentActivityDict[listIndex]['activitiesProfile']['dislikes']['dislikesOnAnnotations']['externalResource']+1
                        if splittedThisActivityStringArray[4] == 'annotation':
                            listOfStudentActivityDict[listIndex]['activitiesProfile']['dislikes']['dislikesOnAnnotations']['externalResource']=listOfStudentActivityDict[listIndex]['activitiesProfile']['dislikes']['dislikesOnAnnotations']['externalResource']+1
                elif splittedThisActivityStringArray[2]=='reply':
                    listOfStudentActivityDict[listIndex]['activitiesProfile']['dislikes']['dislikesOnAnnotations']=listOfStudentActivityDict[listIndex]['activitiesProfile']['dislikes']['likesOnReplies']+1
        
        # Count the un-dislikes to subtract from the dislikes
        elif 'un-disliked' in studentActivitiesListAggregated[listIndex][itemIndex]:
            splittedThisActivityStringArray=  studentActivitiesListAggregated[listIndex][itemIndex].split(' ')
            if  splittedThisActivityStringArray[1] == 'un-disliked':
                if splittedThisActivityStringArray[2]=='annotation':
                    listOfStudentActivityDict[listIndex]['activitiesProfile']['dislikes']['dislikesOnAnnotations']['totalLikes']=listOfStudentActivityDict[listIndex]['activitiesProfile']['dislikes']['dislikesOnAnnotations']['totalDisikes']-1
                    if splittedThisActivityStringArray[3] == 'Note':
                        listOfStudentActivityDict[listIndex]['activitiesProfile']['dislikes']['dislikesOnAnnotations']['note']= listOfStudentActivityDict[listIndex]['activitiesProfile']['dislikes']['dislikesOnAnnotations']['note']-1
                        if splittedThisActivityStringArray[4] == 'annotation':
                            listOfStudentActivityDict[listIndex]['activitiesProfile']['dislikes']['dislikesOnAnnotations']['note']=listOfStudentActivityDict[listIndex]['activitiesProfile']['dislikes']['dislikesOnAnnotations']['note']-1
                    elif splittedThisActivityStringArray[3] == 'Question':
                        listOfStudentActivityDict[listIndex]['activitiesProfile']['dislikes']['dislikesOnAnnotations']['question']= listOfStudentActivityDict[listIndex]['activitiesProfile']['dislikes']['dislikesOnAnnotations']['question']-1
                        if splittedThisActivityStringArray[4] == 'annotation':
                            listOfStudentActivityDict[listIndex]['activitiesProfile']['dislikes']['dislikesOnAnnotations']['question']=listOfStudentActivityDict[listIndex]['activitiesProfile']['dislikes']['dislikesOnAnnotations']['question']-1
                    elif splittedThisActivityStringArray[3] == 'External Resource':
                        listOfStudentActivityDict[listIndex]['activitiesProfile']['dislikes']['dislikesOnAnnotations']['externalResource']= listOfStudentActivityDict[listIndex]['activitiesProfile']['dislikes']['dislikesOnAnnotations']['externalResource']-1
                        if splittedThisActivityStringArray[4] == 'annotation':
                            listOfStudentActivityDict[listIndex]['activitiesProfile']['dislikes']['dislikesOnAnnotations']['externalResource']=listOfStudentActivityDict[listIndex]['activitiesProfile']['dislikes']['dislikesOnAnnotations']['externalResource']-1
                elif splittedThisActivityStringArray[2]=='reply':
                    listOfStudentActivityDict[listIndex]['activitiesProfile']['dislikes']['dislikesOnAnnotations']=listOfStudentActivityDict[listIndex]['activitiesProfile']['dislikes']['likesOnReplies']-1

        # Count the accesses for course, channel, material [username accessed material material_type timestamp]
        elif 'accessed' in studentActivitiesListAggregated[listIndex][itemIndex]:
            listOfStudentActivityDict[listIndex]['activitiesProfile']['access']['totalAccesses'] =listOfStudentActivityDict[listIndex]['activitiesProfile']['access']['totalAccesses']+1
            splittedThisActivityStringArray=  studentActivitiesListAggregated[listIndex][itemIndex].split(' ')
            if splittedThisActivityStringArray[2] == 'course':
                listOfStudentActivityDict[listIndex]['activitiesProfile']['access']['courseAccesses']=listOfStudentActivityDict[listIndex]['activitiesProfile']['access']['courseAccesses']+1
            elif splittedThisActivityStringArray[2] == 'topic':
                listOfStudentActivityDict[listIndex]['activitiesProfile']['access']['topicAccesses']=listOfStudentActivityDict[listIndex]['activitiesProfile']['access']['topicAccesses']+1
            elif splittedThisActivityStringArray[2] == 'channel':
                listOfStudentActivityDict[listIndex]['activitiesProfile']['access']['channelAccesses']=listOfStudentActivityDict[listIndex]['activitiesProfile']['access']['channelAccesses']+1
            elif splittedThisActivityStringArray[2] == 'material':
                if splittedThisActivityStringArray[3]== 'pdf':
                    listOfStudentActivityDict[listIndex]['activitiesProfile']['access']['materialAccesses']['pdfAccess']=listOfStudentActivityDict[listIndex]['activitiesProfile']['access']['materialAccesses']['pdfAccess']+1
                elif splittedThisActivityStringArray[3]== 'video':
                    listOfStudentActivityDict[listIndex]['activitiesProfile']['access']['materialAccesses']['videoAccess']=listOfStudentActivityDict[listIndex]['activitiesProfile']['access']['materialAccesses']['videoAccess']+1

        # PDFs completed
        elif 'completed pdf' in studentActivitiesListAggregated[listIndex][itemIndex]:
            listOfStudentActivityDict[listIndex]['activitiesProfile']['materialProfile']['pdf'][
                'completed'] = listOfStudentActivityDict[listIndex]['activitiesProfile']['materialProfile']['pdf']['completed']+1
        
        # Videos completed
        elif 'completed video' in studentActivitiesListAggregated[listIndex][itemIndex]:
            listOfStudentActivityDict[listIndex]['activitiesProfile']['materialProfile']['video'][
                'completed'] = listOfStudentActivityDict[listIndex]['activitiesProfile']['materialProfile']['video']['completed']+1
            

        # Viewed slide [username viewed slide slide_number timestamp]
        elif 'viewed slide' in studentActivitiesListAggregated[listIndex][itemIndex]:
            listOfStudentActivityDict[listIndex]['activitiesProfile']['materialProfile']['pdf']['slidesViewed']=listOfStudentActivityDict[listIndex]['activitiesProfile']['materialProfile']['pdf']['slidesViewed']+1

            splittedThisActivityStringArray=  studentActivitiesListAggregated[listIndex][itemIndex].split(' ')
            if splittedThisActivityStringArray[3]=='0':
                listOfStudentActivityDict[listIndex]['activitiesProfile']['materialProfile']['pdf']['started']=listOfStudentActivityDict[listIndex]['activitiesProfile']['materialProfile']['pdf']['started']+1

        ####### Timestamps questions
        # Timestamps for videos watched 

        # Played/Paused Videos [username played/paused video video_id duration(seconds) timestamp]
        elif 'played video' in studentActivitiesListAggregated[listIndex][itemIndex] or 'paused video' in studentActivitiesListAggregated[listIndex][itemIndex]:
            splittedThisActivityStringArray=  studentActivitiesListAggregated[listIndex][itemIndex].split(' ')
            # add only the video id in the list of videos played or paused
            for i in range(len(allVideos)): 
                    # If video id is new in the list, create new object of video
                    if(allVideos[i]['videoId'] != splittedThisActivityStringArray[3]):
                        allVideos.append(copy.deepcopy(video))
                        allVideos[i]['videoId']= splittedThisActivityStringArray[3]
                        if('played' in splittedThisActivityStringArray[1]):
                            allVideos[i]['playedDurations']=allVideos[i]['playedDurations']+int(splittedThisActivityStringArray[4])
                        elif('paused' in splittedThisActivityStringArray[1]):
                            allVideos[i]['pausedDurations']=allVideos[i]['pausedDurations']+int(splittedThisActivityStringArray[4])   
                    # If video id is already in the list
                    elif(allVideos[i]['videoId'] == splittedThisActivityStringArray[3]): 
                        if('played' in splittedThisActivityStringArray[1]):
                            allVideos[i]['playedDurations']=allVideos[i]['playedDurations']+int(splittedThisActivityStringArray[4])
                        elif('paused' in splittedThisActivityStringArray[1]):
                            allVideos[i]['pausedDurations']=allVideos[i]['pausedDurations']+int(splittedThisActivityStringArray[4]) 
            # Subtracting played duration from paused duration to find the total time spent in allVideos[i]


            # Started videos
            if splittedThisActivityStringArray[4]=='0': # Meaning that the video is played from the beginning
                    startedVideosIDList = startedVideosIDList.append(splittedThisActivityStringArray[3])
            # Pauses in videos
            elif('paused' in splittedThisActivityStringArray[1]):
                listOfStudentActivityDict[listIndex]['activitiesProfile']['materialProfile']['video']['paused']=listOfStudentActivityDict[listIndex]['activitiesProfile']['materialProfile']['video']['paused']+1
               

    #Finding the unique IDs of videos started to count started videos
    listOfStudentActivityDict[listIndex]['activitiesProfile']['materialProfile']['video']['started']=len(set(startedVideosIDList))

    # Timespent on videos
    timeSpentonVideosInSeconds=0
    for i in range(len(allVideos)):
        timeSpentonVideosInSeconds=timeSpentonVideosInSeconds+(allVideos[i]['playedDurations']-allVideos[i]['pausedDurations'])

    listOfStudentActivityDict[listIndex]['activitiesProfile']['materialProfile']['video']['timespent']=timeSpentonVideosInSeconds
writeToJsonFile(listOfStudentActivityDict)
convertToCsv.export_to_csv()
""" for student in listOfStudentActivityDict:
    writeToJsonFile(student)
    print(student) """

##########################################################################
# At this point, we have our list if dictionary ready to be clustered
##########################################################################