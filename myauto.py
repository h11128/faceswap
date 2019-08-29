import os
import sys
import shutil
import re
import time
import datetime
import subprocess
from tqdm import tqdm
from os.path import split
from tqdm import trange
from time import sleep

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

class Auto():
    def __init__(self):
        self.e1_parameters = '-D mtcnn -A fan -min 64 -een 5 -s -sf -si 500'
        self.e2_parameters = '-D cv2-dnn -A fan -min 64 -een 1 -s -sf -si 500 -sp'
        self.c_parameter = '-c color-transfer -sc sharpen -M predicted -j 10 -k'

    def get_path(self, name = '', video_file_name = '', name2 = '', video_path = ''):
        # Use person name to define the path of all files created during the
        # faceswap process
        # Because the faceswap target celebrity so it's suitable to have names
        # name1 is the name of the video, name2 is the target person's name

        if video_path != '':
            data = split(video_path)
            if video_file_name == '':
                video_file_name = data[-1]
            if name == '':
                name = split(data[0])[-1]
                all_video_path = split(data[0])[0] + '/'

        self.name = name
        self.target_name = name2
        self.video_file_name = video_file_name
        self.all_video_path = 'src/videos/'
        self.src_video_path = self.all_video_path + name +  '/'
        self.src_image_path = 'src/' + name + '/'
        self.target_image_path = 'src/' + name2 +  '/'
        self.know_faces_path = 'faces/known_faces/'

        video_name, file_extend = os.path.splitext(video_file_name)
        self.video_name = video_name
        self.video_file_path = self.src_video_path + video_file_name
        self.extract_faces_path = 'faces/' + name + '/' + video_name + '/'
        self.all_model_path = 'models/'
        self.model_path = 'models/' + name + '_' + name2 + '_model/'
        self.alignment_path = os.path.join(self.all_video_path + name +  '/', video_name + '_alignments.json')
        self.ala_path = os.path.join(self.all_video_path + name +  '/', name + '_alignments.json')
        self.alb_path = os.path.join(self.all_video_path + name2 +  '/', name2 + '_alignments.json')
        self.convert_frame_path = 'converted/' + name + '/' + video_name + '/'
        self.convert_video_path = 'converted/video/' + name + '/' + video_name + '.mp4'
        self.extract_faces_alignment_path = os.path.join('faces/' + name + '/', name + '_' + video_name + '_alignments.json')
        self.any_path = [self.name, self.target_name, self.src_video_path, self.all_video_path,
            self.src_image_path,  self.target_image_path, self.know_faces_path, self.video_name,
            self.video_file_name, self.video_file_path, self.extract_faces_path, self.all_model_path, self.model_path,
            self.alignment_path, self.ala_path, self.alb_path, self.convert_frame_path,
            self.convert_video_path, self.extract_faces_alignment_path]

        return self.any_path

    def download(self, url, path = '', name = ''):
        #format = '-format=mp4'
        # use you-get to download mp4 format videos from youtuebe according to url
        if path == '': path = self.all_video_path
        if name == '': name = self.name

        starting_time =  time.time()
        tqdm.write('Start Downloding %s'%(url))
        download_path = os.path.join(path, name)
        os.system(
            'you-get -o {} {}'.format( download_path, url),)

        time_taken = time.time() - starting_time
        tqdm.write('Finished downloading in {}'.format(
            str(datetime.timedelta(0, time_taken))))

    def get_namelist(self):
        # under the video file, get all names
        file_list = os.listdir(self.all_video_path)
        res = []
        for i in range(len(file_list)):
            if '_' in file_list[i] and "reco" not in file_list[i]:
                res.append(file_list[i])
        return res

    def rename_videos(self):
        # rename the downloaded video to a, b, c ,d and so on. So that it is easy
        # to manage
        src_video_path = self.src_video_path
        file_list = os.listdir(src_video_path)
        k = 0
        for i in range(len(file_list)):
            video_file_path = os.path.join(src_video_path, file_list[i])
            file_name, file_extend = os.path.splitext(file_list[i])
            if len(file_name)>= 2 and (file_extend == '.webm' or file_extend == ".mp4"):
                new_name = chr(k+97) + file_extend
                k += 1
                newfile_path = os.path.join(src_video_path, new_name)
                shutil.copyfile(video_file_path, newfile_path)
                print("From file:", video_file_path, " copy to file: ", newfile_path)
        print("\n")

    def extract(self, video_file_path, output_path, alignment_path, e_parameters):
        # run extract command to extract faces from a video. One should specify an alignment
        # file. This function read the result of command and keep missing faces frames name
        # during running.

        missing_txt = 'reco/' + self.name + '_missing_' + self.video_file_name + '.txt'
        Missing = False

        starting_time =  time.time()

        tqdm.write('\033[92mStart Extracting %s \033[00m'%(video_file_path))
        alignment = '-al {}'.format(alignment_path)
        if alignment_path == '':
            alignment = '-al {}'.format(self.ala_path)
        command = 'python faceswap.py extract -i {} -o {} {} {}'.format(
            video_file_path, output_path, alignment, e_parameters)
        print("\033[95mCommand:\033[00m" , command)
        procExe = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, universal_newlines=True)
        i = 0
        while procExe.poll() is None:
            line = procExe.stdout.readline()

            if Missing == False and "Unable to open image" in line:
                Missing = True
                print("Find missing image. Keep Missing image!!!!!!!!!!!!!!!! ")
                f = open(missing_txt,'w')
                f.write(line)
            if Missing and "Unable to open image" in line:
                f.write(line)
            if 'Running' in line:
                print(line.strip('\n'), end="\r", flush=True)
            elif line.strip('\n').strip(' ') == '':
                continue
            else:
                print(line.strip('\n')+ '                                    ')
        f.close()
        time_taken = time.time() - starting_time

        tqdm.write('\033[92mFinished Extracting in {}. \n\033[00m'.format(
            str(datetime.timedelta(0, time_taken))))

    def recognition(self, image_path, know_faces_path, overwrite = False, keep_temp = False, to_path = 'reco/', keep_False = False):
        # This function utilize face_recognition to help delete false positive faces during extracting.
        # overwrite means whether to start a new recognition when there are already a recognition record.
        # keep_temp means whether to keep the copy images file for debugging
        # to_path is where the reco_txt and temp copy image goes
        # keep_False means whether to keep False positives in original extract image path
        # you should put the person's image to the know_face_path so that it can recognize according to this image
        name = self.name
        data = split(split(image_path)[0])
        if data[-1] == name:
            reco_name = name + '_whole'
        else:
            reco_name = 'extract_' + name + "_" + data[-1]
        reco_txt = to_path + reco_name + '.txt'
        reco_path = to_path + reco_name + '_reco/'

        starting_time =  time.time()
        tqdm.write("\033[92mBegin recognition of {}'s image in {} \033[00m".format(name, image_path))
        if not os.path.isfile(reco_txt) or overwrite:
            print("Not recognition file found! \nStart new recognition of %s 's faces to %s"%(name, reco_txt))
            if overwrite and os.path.isfile(reco_txt):
                os.remove(reco_txt)
            f = open(reco_txt,'w')
            i = 0
            image_number = len(os.listdir(image_path))

            command = 'face_recognition --cpus 1 %s %s'% (know_faces_path, image_path)
            print("\033[95mCommand: \033[00m" , command)
            procExe = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, universal_newlines=True)
            while procExe.poll() is None:
                line = procExe.stdout.readline()
                i+=1
                f.write(line)
                if len(line) > 3 and (i == 2 or i % (image_number//10) == 0 or i == image_number):
                    print('Recognized %s file: '%(i), line.strip('\n') + '                    ')
                #print('Recognized %s file: '%(i), line.strip('\n')+'                          ', end="\r", flush=True)

            f.close()
            time_taken = time.time() - starting_time
            tqdm.write("Finished recognition in {}. \n".format(
                str(datetime.timedelta(0, time_taken))))
        else:
            print("Exist recognition file! Skip recognition \n")

        self.read_recognition(name, reco_txt, reco_path, keep_temp, keep_False)

        time_taken = time.time() - starting_time
        print('\033[92mRecognition finished in {}. \n\033[00m'.format(
            str(datetime.timedelta(0, time_taken))))

    def read_recognition(self, name, reco_txt, reco_path, keep_temp, keep_False):
        # a function help to read the recognition file and delete false positives
        print("Read recognition file")
        f = open(reco_txt, 'r')
        i  = 0
        j = 0
        k = 0
        lines = f.readlines()
        for line in lines:
            line = line.strip('\n')
            data = line.split(',')
            k += 1
            if len(data) == 2:
                reco_person = data[1]
                new_path = reco_path + reco_person + "/"
                os.makedirs(new_path, exist_ok = True)
                file_path = data[0]
                original_file_name = split(file_path)[-1]
                newfile_path = new_path + original_file_name
                if os.path.exists(file_path):
                    shutil.copyfile(file_path, newfile_path)
                    j += 1
                    if data[-1] != name and not keep_False:
                        os.remove(file_path)
                        i +=1
                else:
                    print("Error: File not found but appear at reco_txt: ",  file_path)
            else:
                print("!!!!!!!:", data)

        f.close()
        print("Remove %s false positives! "%(i))
        print('Copy %s files to %s, Correct_copy ='%(j, reco_path), k-1==j)
        if not keep_temp:
            shutil.rmtree(reco_path)
            print("Deleted temporary file.")
        else:
            print("Keep temporary file in %s. "%reco_path)

    def append_missing(self):
        #Append missing file after all frames have been converted.
        name = self.video_name
        missing_txt = 'reco/' + self.name + '_missing_' + self.video_file_name + '.txt'
        f = open(missing_txt, 'r')

        tqdm.write("\033[92mBegin append missing image! \033[00m   ")
        lines = f.readlines()
        cur_number = 0
        numbers = [[]]
        i = 0
        for line in lines:
            data = re.split("[, ' / ]", line.strip('\n'))
            if '.png' in data[-2]:
                number = int(data[-2][len(name)+2:-4])
                if number == cur_number + 1:
                    numbers[i].append(number)
                    cur_number = number
                else:
                    i += 1
                    numbers.append([number])
                    cur_number = number
        if numbers[0] == [1] and 1 in numbers[1]:
            numbers.remove([1])
        f.close()
        print(numbers)

        def return_file_name(number, video_name = name , file_length = len(data[-2])):
            need = [video_name + '_']
            number_of_0 = file_length -5-len(video_name)-len(str(number))
            for i in range(number_of_0):
                need.append('0')
            need.append(str(number))
            need.append('.png')
            need = ''.join(need)
            return need

        file_length = len(data[-2])
        for i in range(len(numbers)):
            target_name = return_file_name(numbers[i][-1]+1)
            need_path = os.path.join(self.convert_frame_path,target_name)
            if not os.path.exists(need_path):
                target_name = return_file_name(numbers[i][0]-1)
                need_path = os.path.join(self.convert_frame_path,target_name)
            if not os.path.exists(need_path):
                print("Not available image found between %s and %s!!! Skip appending %s"%(
                return_file_name(numbers[i][-1]+1), return_file_name(numbers[i][0]-1)), numbers[i]
                    )
            else:
                #print("Append (", end='')
                for j in numbers[i]:
                    file_name = return_file_name(j)
                    target_path = os.path.join(self.convert_frame_path,file_name)
                    shutil.copyfile(need_path, target_path)
                    print("Append %s from %s "%(file_name, target_name))
                #print(") from file %s"%(need_path))

        tqdm.write("\033[92mFinish appending!!! \n\033[00m")

    def check_faces(self, alignment_path, face_image_path):
        # check whether the face image does not fit the file in alignment_path
        tqdm.write("\033[92mBegin check %s's faces with alignment file.\033[00m"%(self.name))
        command1 = 'python tools.py alignments -j remove-faces -a {} -fc {} -o console'.format(
            alignment_path, face_image_path)
        command2 = 'python tools.py alignments -j leftover-faces -a {} -fc {} -o console'.format(
            alignment_path, face_image_path)
        print("\033[95mCommand: \033[00m" , command1)
        os.system(command1)
        print("\033[95mCommand: \033[00m" , command2)
        output = os.popen(command2).read()
        print(output, end = '')
        if "No faces were found meeting the criteria" not in output:
            raise ValueError("Find leftover faces in the alignment file that does \
                not exist in image file. Check the image file to see what's wrong")

        tqdm.write("\033[92mFinished checking %s's images!!!!!. \n\033[00m"%(self.name))

    def merge_check_faces(self, overwrite_merge = False):
        # merge alignment file, and check if there are leftover or remove image with this alignment
        starting_time =  time.time()

        tqdm.write("\033[92mBegin merge and remove alignment file!\033[00m")

        standard_name_length = len('a_alignments')
        src_video_path = self.src_video_path
        image_path = self.src_image_path
        name = self.name

        file_list = os.listdir(src_video_path)
        print(file_list)
        alignment_paths = []
        whole_alignment = ''
        merged_exist = False
        # get filelist, find already exist alignment, get all alignment file
        for i in range(len(file_list)):
            file_name, file_extend = os.path.splitext(file_list[i])
            if file_extend == '.json':
                if name in file_name:
                    merged_exist = True
                    whole_alignment = os.path.join(src_video_path, file_list[i])
                if len(file_name) == standard_name_length:
                    alignment_paths.append(os.path.join(src_video_path, file_list[i]))

        if merged_exist and not overwrite_merge:
            print("Already found %s 's merged alignment file %s!"%(name, whole_alignment))
        else:
            print(alignment_paths)
            if len(alignment_paths) > 1:
                for i in alignment_paths:
                    whole_alignment = whole_alignment + ' ' + i
                command = 'python tools.py alignments -j merge -a {} -fc {} -o console'.format(
                    whole_alignment, image_path)
                print("\033[95mCommand: \033[00m" , command)
                output = os.popen(command).read()
                print(output)
                words = output.split("\n")
                target = ''
                for line in words:
                    if 'Writing' in line:
                        target = line.strip("''")
                        break
                data = re.split(r'[, /]', target)
                data = split(data[-1])
                new_alignment = data[-1]
                whole_alignment = os.path.join(src_video_path, new_alignment)

            elif len(alignment_paths) == 1:
                print('Only 1 valid alignment file. No need to merge.')
                whole_alignment = alignment_paths[0]
            else:
                raise ValueError('Error! Not Alignment Files Found!!')

            print(name, "'s alignment file has been merge to ", whole_alignment)
            self.check_faces(whole_alignment, image_path)

        new_alignment = self.ala_path
        shutil.copyfile(whole_alignment, new_alignment)

        time_taken = time.time() - starting_time
        tqdm.write("\033[92mFinished merging the alignment file. From path: {} to path: {} in {}. \n\033[00m".format(
            whole_alignment, new_alignment, str(datetime.timedelta(0, time_taken))))

    def sort_faces(self, input_path, output_path, parameters = '-fp rename -s hist -g hist'):
        # sort faces according to histogram similarity
        print("Sorting faces of " + self.name)
        command_sort = 'python tools.py sort -i "{}" -o {} {} '.format(
            input_path, output_path, parameters)
        print("\033[95mCommand: \033[00m" , command_sort)
        os.system(command_sort)

    def extract_videos_to_source_images(self, video_path = '', know_faces_path = '', want_recognition = True):
        # This can be used to extract muliiple videos of the same person to gather
        # different images for this person to train a model
        output_path = self.src_image_path
        src_video_path = self.src_video_path
        file_list = os.listdir(src_video_path)

        starting_time =  time.time()

        tqdm.write('\033[92mStart extract_videos_to_source_images!\033[00m')
        if os.path.isfile(video_path):
            output_path = self.extract_faces_path
            self.extract(video_path, output_path, self.extract_faces_alignment_path, self.e2_parameters)
        else:
            for i in range(len(file_list)):
                file_name, file_extend = os.path.splitext(file_list[i])
                video_path = os.path.join(src_video_path, file_list[i])
                alignments_path = os.path.join(src_video_path, file_name + '_alignments.json')
                if len(file_name) <= 2 and (file_extend == '.webm' or file_extend == ".mp4"):
                    self.extract(video_path, output_path, alignments_path, self.e1_parameters)

        self.sort_faces(output_path, output_path)

        if know_faces_path == 'Default':
            know_faces_path = self.know_faces_path

        if os.path.exists(know_faces_path) and want_recognition:
            self.recognition(output_path, know_faces_path, overwrite = True, keep_temp = False)
            if os.path.isfile(video_path):
                self.check_faces(self.extract_faces_alignment_path, output_path)
            else:
                self.merge_check_faces(overwrite_merge = False)
        time_taken = time.time() - starting_time

        tqdm.write('\033[92mFinished extract_videos_to_source_images in {}. \n\033[00m'.format(
            str(datetime.timedelta(0, time_taken))))

    def train(self, model_name = 'villain', save_iteration = 5000, batch_size = 32, iterations = 30000, warp_to_landmarks = False):
        # train a model, the model_name, save_iteration, batch_size, itrations, warp_to_landmars
        # are some of the most important parameters for the train process.

        starting_time =  time.time()

        tqdm.write('\033[92mStart training with {} to {} model\033[00m'.format(
            self.name, self.target_name))

        #-ag -wl
        t_parameters = '-t {} -s {} -bs {} -it {} -ag -msg -o'.format(
            model_name, save_iteration, batch_size, iterations
            )
        if warp_to_landmarks :
            t_parameters += ' -wl'
        command_train = 'python faceswap.py train -A {} -B {} -ala {} -alb {} -m {} {} '.format(
            self.src_image_path, self.target_image_path, self.ala_path, self.alb_path, self.model_path, t_parameters)
        print("\033[95mCommand: \033[00m" , command_train)
        os.system(command_train)

        time_taken = time.time() - starting_time

        tqdm.write('\033[92mFinished training in {}. \n\033[00m'.format(
            str(datetime.timedelta(0, time_taken))))

    def convert_to_frames(self, model_path = ''):
        # convert video to frames
        starting_time =  time.time()
        tqdm.write("\033[92mStart converting {}'s faces to {}'s faces in frames\033[00m".format(
            self.name, self.target_name))
        c_parameter = self.c_parameter

        # Find Model
        swap_path = 'models/' + self.target_name + '_' + self.name + '_model/'
        if os.path.exists(self.model_path):
            model_path = self.model_path
        elif os.path.exists(swap_path):
            model_path = swap_path
            self.model_path = swap_path
            c_parameter += ' -s'
        elif os.path.exists(model_path):
            model_path = model_path
            index = model_path.find(self.target_name)
            if index < 8:
                c_parameters += ' -s'
        else:
            #self.train(model_name = 'villain', save_iteration = 1000, batch_size = 32, iterations = 20000)
            raise ValueError('Model not found. Please check your model directory and rename your mdoel!!!!')

        command1 = 'python faceswap.py convert -i {} -o {} -al {} -m {} {}'.format(
            self.video_file_path , self.convert_frame_path, self.extract_faces_alignment_path, model_path, c_parameter)
        print("\033[95mCommand: \033[00m" , command1)
        os.system(command1)
        time_taken = time.time() - starting_time
        tqdm.write("\033[92mImage converted in {}!! Find it at {}. \n \033[00m".format(
            str(datetime.timedelta(0, time_taken)), self.convert_video_path))

        missing_txt = 'reco/' + self.name + '_missing_' + self.video_file_name + '.txt'
        if os.path.exists(missing_txt):
            self.append_missing()

    def convert(self, video_file_name, target_name, skip_extracting = True, know_faces_path = 'Default', model_path = ''):
        # The whole convert process
        # given a video file and a target name, the output will be a converted video
        # noticed that you have to have a model which is the name_target_name model otherwise you have to give the model_path
        # otherwise it will start to train model by itself.
        starting_time =  time.time()
        tqdm.write("\033[92mStart converting {}'s video to {}'s video\033[00m".format(
            self.name, self.target_name))

        if not os.path.exists(self.extract_faces_alignment_path) or skip_extracting == False:
            self.extract_videos_to_source_images(self.video_file_path, know_faces_path)
        else:
            print("Skip extract_videos_to_source_images")

        self.convert_to_frames(model_path = '')

        if not os.path.exists('converted/video/' + self.name + '/'):
            os.makedirs('converted/video/' + self.name + '/')
        command2 = 'python tools.py effmpeg -a gen-vid -i {} -o {} -r {} -fps -1 -m'.format(
            self.convert_frame_path, self.convert_video_path, self.video_file_path)
        print("\033[95mCommand: \033[00m" , command2)
        os.system(command2)

        time_taken = time.time() - starting_time
        tqdm.write("\033[92mVideo converted in {}!! Find it at {}. \n \033[00m".format(
            str(datetime.timedelta(0, time_taken)), self.convert_video_path))

    def swap_from_two_video(self, video1, video2, know_faces_path = 'False'):
        # given that there is not model, swap the faces of video1 to video2
        b = Auto()
        b.get_path(video_path = video2)
        self.get_path(name2 = b.name, video_path = video1)

        self.src_image_path = self.extract_faces_path
        self.target_image_path = b.extract_faces_path
        self.ala_path = self.extract_faces_alignment_path
        self.alb_path = b.extract_faces_alignment_path
        if not os.path.exists(know_faces_path):
            know_faces_path = self.know_faces_path
        self.extract_videos_to_source_images(video1, know_faces_path)
        b.extract_videos_to_source_images(video2, know_faces_path)

        #self.train(model_name = 'villain', save_iteration = 1000, batch_size = 32, iterations = 20000)
        self.convert(self.video_file_name, self.target_name, True)

    def debug(self, video_name, target_name):
        src_name = self.name
        src_video_path = self.src_video_path
        extract_faces_path = self.extract_faces_path
        convert_frame_path = self.convert_frame_path
        convert_video_path = self.convert_video_path
        extract_faces_alignment_path = self.extract_faces_alignment_path
        video_file_path = self.video_file_path
        c_parameter = self.c_parameter

        swap = False
        swap_path = 'models/' + target_name + '_' + src_name + '_model/'
        if os.path.exists(self.model_path):
            model_path = self.model_path
            print("Find model in %s"%(model_path))
        elif os.path.exists(swap_path):
            model_path = swap_path
            self.model_path = swap_path
            swap = True
            print("Find model in %s"%(model_path))
        else:
            raise ValueError('Model not found. Please check your model directory and rename your mdoel!!!!')

        print('python faceswap.py convert -i {} -o {} -al {} -m {} {}'.format(
            video_file_path, convert_frame_path, extract_faces_alignment_path, model_path, c_parameter))

        os.system('python faceswap.py convert -i {} -o {} -al {} -m {} {}'.format(
            video_file_path, convert_frame_path, extract_faces_alignment_path, model_path, c_parameter))

    def debug2(self, video, image_path):
        data = split(split(image_path)[0])


urllist = ['https://www.youtube.com/watch?v=DXy3S2WkyoE',
            'https://www.youtube.com/watch?v=u32stvz-Dk8',
            'https://www.youtube.com/watch?v=K27CKdkGUZ4',
            'https://www.youtube.com/watch?v=6Rjn1PMd0-U']


namelist = ['Hillary_Clinton',
            'Jimmy_Fallon',
            'Kate_McKinnon']
target_name = ['Donald_Trump',
                'Donald_Trump',
                'Hillary_Clinton']
video_file_list = ['c.mp4',
                    'a.mp4',
                    'a.mp4']
a = Auto()

for i in range(2,3):
    #a.get_path('Hillary_Clinton','a.webm', 'Jimmy_Fallon')
    #a.extract_videos_to_source_images()

    #a.train(model_name = 'villain', save_iteration = 1000, batch_size = 32, iterations = 20000)

    a.get_path(name2 = 'Hillary_Clinton', video_path = 'src/videos/Donald_Trump/c.mp4')
    a.convert(a.video_file_name, a.target_name, skip_extracting = True, know_faces_path = 'Default', model_path = '')

    #a.append_missing()
    #a.train(model_name = 'villain', save_iteration = 1000, batch_size = 32, iterations = 20000)
    #a.recognition(a.extract_faces_path, a.know_faces_path, overwrite = False, keep_temp = False)
    #a.sort_faces(parameters = '-fp rename -k -s face -g hist')
    #a.check_faces(a.ala_path, a.src_image_path)
