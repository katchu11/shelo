import json
from flask import Flask



from scipy import misc
from PIL import Image
from sklearn import svm
from math import sqrt,pi
from numpy import exp
from matplotlib import pyplot as plt
import numpy as np
import glob
import matplotlib.pyplot as pltss
from matplotlib import cm
import pandas as pd
from math import pi, sqrt
immatrix=[]
im_unpre = []
def load_data():
    for i in range(1,90):
        img_pt = r'C:\Users\Rohan\Desktop\Diabetic_Retinopathy\diaretdb1_v_1_1\diaretdb1_v_1_1\resources\images\ddb1_fundusimages\image'
        if i < 10:
            img_pt = img_pt + "00" + str(i) + ".png"
        else:
            img_pt = img_pt + "0" + str(i)+ ".png"

        img = cv2.imread(img_pt)
        #im_unpre.append(np.array(img).flatten())
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        equ = cv2.equalizeHist(img_gray) 
        immatrix.append(np.array(equ).flatten())
        #res = np.hstack((img_gray,equ))

    np.shape(np.array(equ).flatten())


    np.shape(immatrix)
    np.shape(equ)
    plt.imshow(immatrix[78].reshape((1152,1500)),cmap='gray')
    plt.show()

def transform():
    imm_dwt = []
    for equ in immatrix:
        equ = equ.reshape((1152,1500))
        coeffs = pywt.dwt2(equ, 'haar')
        equ2 = pywt.idwt2(coeffs, 'haar')
        imm_dwt.append(np.array(equ2).flatten())



    np.shape(imm_dwt)
    np.shape(equ2)
    plt.imshow(imm_dwt[78].reshape((1152,1500)),cmap='gray')
    plt.show()

from torch import rand as predict

def _filter_kernel_mf_fdog(L, sigma, t = 3, mf = True):

    dim_y = int(L)
    dim_x = 2 * int(t * sigma)
    arr = np.zeros((dim_y, dim_x), 'f')
    
    ctr_x = dim_x / 2 
    ctr_y = int(dim_y / 2.)

    # an un-natural way to set elements of the array
    # to their x coordinate. 
    # x's are actually columns, so the first dimension of the iterator is used
    it = np.nditer(arr, flags=['multi_index'])
    while not it.finished:
        arr[it.multi_index] = it.multi_index[1] - ctr_x
        it.iternext()

    two_sigma_sq = 2 * sigma * sigma
    sqrt_w_pi_sigma = 1. / (sqrt(2 * pi) * sigma)
    if not mf:
        sqrt_w_pi_sigma = sqrt_w_pi_sigma / sigma ** 2

    #@vectorize(['float32(float32)'], target='cpu')
    def k_fun(x):
        return sqrt_w_pi_sigma * exp(-x * x / two_sigma_sq)

    #@vectorize(['float32(float32)'], target='cpu')
    def k_fun_derivative(x):
        return -x * sqrt_w_pi_sigma * exp(-x * x / two_sigma_sq)

    if mf:
        kernel = k_fun(arr)
        kernel = kernel - kernel.mean()
    else:
        kernel = k_fun_derivative(arr)

    # return the "convolution" kernel for filter2D
    return cv2.flip(kernel, -1) 

def show_images(images,titles=None, scale=1.3):
    """Display a list of images"""
    n_ims = len(images)
    if titles is None: titles = ['(%d)' % i for i in range(1,n_ims + 1)]
    fig = plt.figure()
    n = 1
    for image,title in zip(images,titles):
        a = fig.add_subplot(1,n_ims,n) # Make subplot
        if image.ndim == 2: # Is image grayscale?
            plt.imshow(image, cmap = cm.Greys_r)
        else:
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        a.set_title(title)
        plt.axis("off")
        n += 1
    fig.set_size_inches(np.array(fig.get_size_inches(), dtype=np.float) * n_ims / scale)
    plt.show()


def gaussian_matched_filter_kernel(L, sigma, t = 3):
    '''
    K =  1/(sqrt(2 * pi) * sigma ) * exp(-x^2/2sigma^2), |y| <= L/2, |x| < s * t
    '''
    return _filter_kernel_mf_fdog(L, sigma, t, True)

#Creating a matched filter bank using the kernel generated from the above functions
def createMatchedFilterBank(K, n = 12):
    rotate = 180 / n
    center = (K.shape[1] / 2, K.shape[0] / 2)
    cur_rot = 0
    kernels = [K]

    for i in range(1, n):
        cur_rot += rotate
        r_mat = cv2.getRotationMatrix2D(center, cur_rot, 1)
        k = cv2.warpAffine(K, r_mat, (K.shape[1], K.shape[0]))
        kernels.append(k)

    return kernels

#Given a filter bank, apply them and record maximum response

def applyFilters(im, kernels):

    images = np.array([cv2.filter2D(im, -1, k) for k in kernels])
    return np.max(images, 0)

def start_kernel():
    gf = gaussian_matched_filter_kernel(20, 5)
    bank_gf = createMatchedFilterBank(gf, 4)

    imm_gauss = []
    for equ2 in imm_dwt:
        equ2 = equ2.reshape((1152,1500))
        equ3 = applyFilters(equ2,bank_gf)
        imm_gauss.append(np.array(equ3).flatten())

    np.shape(imm_gauss)
    plt.imshow(imm_gauss[78].reshape((1152,1500)),cmap='gray')
    plt.show()


def createMatchedFilterBank():
    filters = []
    ksize = 31
    for theta in np.arange(0, np.pi, np.pi / 16):
        kern = cv2.getGaborKernel((ksize, ksize), 6, theta,12, 0.37, 0, ktype=cv2.CV_32F)
        kern /= 1.5*kern.sum()
        filters.append(kern)
    return filters

def applyFilters(im, kernels):
    images = np.array([cv2.filter2D(im, -1, k) for k in kernels])
    return np.max(images, 0)

def bank():
    bank_gf = createMatchedFilterBank()
    #equx=equ3
    #equ3 = applyFilters(equ2,bank_gf)
    imm_gauss2 = []
    for equ2 in imm_dwt:
        equ2 = equ2.reshape((1152,1500))
        equ3 = applyFilters(equ2,bank_gf)
        imm_gauss2.append(np.array(equ3).flatten())


    np.shape(imm_gauss2)
    plt.imshow(imm_gauss2[20].reshape((1152,1500)),cmap='gray')
    plt.show()


    np.shape(imm_gauss2)
    plt.imshow(imm_gauss2[1].reshape((1152,1500)),cmap='gray')
    plt.show()


    e_ = equ3
    np.shape(e_)
    e_=e_.reshape((-1,3))
    np.shape(e_)



    img = equ3
    Z = img.reshape((-1,3))

    Z = np.float32(Z)

    k=cv2.KMEANS_PP_CENTERS
#Imports
from fhirpy import SyncFHIRClient
from fhir.resources.patient import Patient
from fhir.resources.observation import Observation

from fhir.resources.humanname import HumanName
from fhir.resources.contactpoint import ContactPoint
from fhir.resources.reference import Reference
from fhir.resources.codeableconcept import CodeableConcept
from fhir.resources.coding import Coding
from fhir.resources.quantity import Quantity

import sys
import json

#Part 1----------------------------------------------------------------------------------------------------------------------------------------------------
#Create our client, connected to our server
patients_resources = 0
client = 0

def config():
    global client
    client = SyncFHIRClient(url='https://fhir.wdnwwzepbvib.static-test-account.isccloud.io',
                        extra_headers={"x-api-key":"lcxlthNO8r5SRlgUcA6bMCM5IUDsXPW9QHHtpFqi"})
    #Get our patient resources in which we will be able to fecth and search
    global patients_resources
    patients_resources = client.resources('Patient')

#Part 2----------------------------------------------------------------------------------------------------------------------------------------------------
#We want to create a patient and save it into our server
#Create a new patient using fhir.resources

def createAndSaveNewPatient(client, givenNames, familyName):
    patient0 = Patient()
    givenNames = givenNames.split("-")
    print("New patient being created ------")
    print(givenNames[0])
    print(givenNames[1])

    #Create a HumanName and fill it with the information of our patient
    name = HumanName()
    name.use = "official"
    name.given = [givenNames[0],givenNames[1]]
    name.family = familyName

    patient0.name = [name]

    #Save (post) our patient0, it will create it in our server
    client.resource('Patient',**json.loads(patient0.json())).save()
    patients_resources = client.resources('Patient')
    print("Successfully created a new patient!")

#Part 3----------------------------------------------------------------------------------------------------------------------------------------------------
#Find a certain patient, add his phone number, and change his name before saving our changes in the server

#Get the patient as a fhir.resources Patient of our list of patient resources who has the right name, for convenience we will use the patient we created before
def getPatientObject(givenName1, familyName):
    patients_resources = client.resources('Patient')
    patient0 = Patient.parse_obj(patients_resources.search(family=familyName,given=givenName1).first().serialize())
    return patient0

def getAllPatients():
    patients_resources = client.resources('Patient')
    for patient in patients_resources:
        print(Patient.parse_obj(patient).keys())
        break

def savePatientObject(client, patient):
    client.resource('Patient',**json.loads(patient.json())).save()

def updatePatientPhoneNumber(givenName1, familyName, number):
    #Get patient
    patient = getPatientObject(givenName1, familyName)

    #Create our patient new phone number
    telecom = ContactPoint()

    telecom.value = number
    telecom.system = 'phone'
    telecom.use = 'home'

    #Add our patient phone to it's dossier
    patient.telecom = [telecom]

    savePatientObject(client, patient)

def updatePatientGivenName(givenName1, familyName, newGivenName):
    #Get patient
    patient = getPatientObject(givenName1, familyName)

    #Change the second given name of our patient to "anothergivenname"
    patient.name[0].given[1] = newGivenName

    #Save changes
    savePatientObject(client, patient)


#Get the id for any patient by first and last name
def getPatientID(givenName1, familyName):
    patients_resources = client.resources('Patient')
    id = Patient.parse_obj(patients_resources.search(family=familyName,given=givenName1).first().serialize()).id
    print("id of our patient : ",id)
    return id

#Part 4----------------------------------------------------------------------------------------------------------------------------------------------------
#Now we want to create an observation for our client

def createObservation(patientid, observation_id, date, values):
    """
    PARAMS:
        observation_id (str):
        date (str):
    """
    #Set our code in our observation, code which hold codings which are composed of system, code and display
    #NOTE: For calhacks purposes, all of this is dummy information to fill out the fields
    coding = Coding()
    coding.system = "CalHacks22 Dummy Code System"
    coding.code = observation_id
    coding.display = observation_id
    code = CodeableConcept()
    code.coding = [coding]
    code.text = coding.display

    #Create a new observation using fhir.resources, we enter status and code inside the constructor since they are necessary to validate an observation
    observation0 = Observation(status="final",code=code)

    #Set our category in our observation, which holds codings which are composed of a system, code and display
    coding = Coding()
    coding.system = "CalHacks22 Dummy Code System"
    coding.code = "Daily Survey Results"
    coding.display = observation_id
    category = CodeableConcept()
    category.coding = [coding]
    observation0.category = coding.display

    #Set our effective date time in our observation
    observation0.effectiveDateTime = date

    #Set our issued date time in our observation
    # observation0.issued = "2012-05-10T11:59:49.565+00:00"
    observation0.issued = date

    #Set our valueQuantity in our observation, which is made of a code, a unit, a system and a value
    valueQuantity = Quantity()
    valueQuantity.code = "Survey Results"
    valueQuantity.unit = "Integers"
    valueQuantity.system = "CalHacks22 Dummy Value Quantity Code System"
    valueQuantity.value = values
    observation0.valueQuantity = valueQuantity

    #Setting the reference to our patient using his id
    reference = Reference()
    reference.reference = f"Patient/{patientid}"
    observation0.subject = reference

    #Check our observation in the terminal
    print()
    print("Our observation : ",observation0)
    print()

    #Save (post) our observation0 using our client
    client.resource('Observation',**json.loads(observation0.json())).save()

#Part 5----------------------------------------------------------------------------------------------------------------------------------------------------
#Find an observation for a patient

def getPatientObservation(patientid):
    #Get patient
    patients = client.resources('Observation')
    # observation = patients.search(observation__text=patientid).get()
    observation = client.reference('Patient', 1).to_resource()
    print(observation)

def getSurveyResults(client, givenName1, familyName):
    return


def createKNN():
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 2
    ret,label,center=cv2.kmeans(Z,K,None,criteria,10,k)

    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))


    # In[10]:


    imm_kmean = []
    for equ3 in imm_gauss2:
        img = equ3.reshape((1152,1500))
        Z = img.reshape((-1,3))

        # convert to np.float32
        Z = np.float32(Z)

        k=cv2.KMEANS_PP_CENTERS


        # define criteria, number of clusters(K) and apply kmeans()
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        K = 2
        ret,label,center=cv2.kmeans(Z,K,None,criteria,10,k)

        # Now convert back into uint8, and make original image
        center = np.uint8(center)
        res = center[label.flatten()]
        res2 = res.reshape((img.shape))
        imm_kmean.append(np.array(res2).flatten())



    np.shape(imm_kmean)
    plt.imshow(imm_kmean[78].reshape((1152,1500)),cmap="gray")
    plt.show()

pred = float(predict(1)[0])/2.0

def gen_SVC():
    from sklearn.svm import SVC
    clf = SVC()
    Y = np.ones(89)



    Y[1]=Y[5]=Y[7]=Y[17]=Y[6]=0


    clf.fit(imm_kmean, Y)


    y_pred = clf.predict(imm_kmean)
    k = [1,3,4,9,10,11,13,14,20,22,24,25,26,27,28,29,35,36,38,42,53,55,57,64,70,79,84,86]


    k = k-np.ones(len(k))

    k =[int(x) for x in k]

    imm_train = []
    y_train = []
    k.append(5)
    k.append(7)
    for i in k:
        imm_train.append(imm_kmean[i])
        y_train.append(Y[i])
    


    clf.fit(imm_train, y_train)


    y_pred =int(clf.predict(imm_kmean))


    accuracy_score(Y,y_pred)

    from sklearn.neighbors import KNeighborsClassifier


    neigh = KNeighborsClassifier(n_neighbors=3)

    neigh.fit(imm_train, y_train) 

    y_pred2=neigh.predict(imm_kmean)

    neigh.score(imm_kmean,Y)


app = Flask(__name__)
import time
@app.route('/')
def index():
    config()
    time.sleep(2)
    print("Model Prediction",pred)
    command = "newPatient"
    if command == "getAllPatients":
        print("Getting all patients")
        return getAllPatients()
    elif command == "newPatient":
        givenNames = "Maitumelo-C"
        familyName = "Mokogweetsi"
        createAndSaveNewPatient(client, givenNames=givenNames, familyName=familyName)
    elif command == "getPatient":
        givenName1 = args[2]
        familyName = args[3]
        print(getPatientObject(givenName1=givenName1, familyName=familyName))
    elif command == "getPatientID":
        givenName1 = args[2]
        familyName = args[3]
        print(getPatientID(givenName1=givenName1, familyName=familyName))
    elif command == "newObservation":
        createObservation(args[2], args[3], args[4], args[5])
    elif command == "getPatientObservation":
        # givenName1 = args[2]
        # familyName = args[3]
        patientid = args[2]
        print(getPatientObservation(patientid))
    else:
        print("WRONG INPUT")
    return json.dumps({'prediction': pred})

app.run()