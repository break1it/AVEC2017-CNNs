import os
from pydub import AudioSegment
import numpy

if __name__ == '__main__':
    loadpath = 'D:/PythonProjects_Data/AVEC2017_Data/Step1_Filter/'
    savepath = 'D:/PythonProjects_Data/AVEC2017_Data/Step2_VoiceSeparate/'

    for foldA in os.listdir(loadpath):
        for foldB in os.listdir(os.path.join(loadpath, foldA)):
            print('Treating %s %s' % (foldA, foldB))
            os.makedirs(os.path.join(savepath, foldA, foldB))

            voiceData = AudioSegment.from_file(os.path.join(loadpath, foldA, foldB, foldB[0:3] + '_AUDIO.wav'))
            transcription = numpy.genfromtxt(fname=os.path.join(loadpath, foldA, foldB, foldB[0:3] + '_TRANSCRIPT.csv'),
                                             dtype=str, delimiter='\t')
            # voiceData[0:5000].export('Test.wav', format='wav')

            with open(os.path.join(savepath, foldA, foldB, 'Transcription.csv'), 'w') as file:
                file.write('ID,Start Time,Stop Time, Speaker, Transcription\n')
                for index in range(1, len(transcription)):
                    file.write('%04d,%s,%s,%s,%s\n' % (
                        index, transcription[index][0], transcription[index][1], transcription[index][2],
                        transcription[index][3]))

            for index in range(1, len(transcription)):
                # print(int(float(transcription[index][0]) * 1000), int(float(transcription[index][1]) * 1000))
                if transcription[index][2] != 'Participant': continue
                voiceData[int(float(transcription[index][0]) * 1000):int(float(transcription[index][1]) * 1000)].export(
                    os.path.join(savepath, foldA, foldB, 'Speech_%04d.wav' % index), format='wav')
            # exit()
