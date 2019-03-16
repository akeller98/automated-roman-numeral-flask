# This line is a convenience to import most packages you'll need. You may need to import others (e.g. random and cmath)
import IPython, numpy as np, scipy as sp, matplotlib.pyplot as plt, matplotlib, sklearn, librosa, cmath,math, csv
from chord_training import training_chord_set, training_chord_labels,chord_dictionary,chord_vectors,ks_key_set,ks_key_labels,key_dictionary,key_index_dictionary,root_index_dictionary, major_roman_dictionary,minor_roman_dictionary
import sys

def load_audio(audio_path):
    """
    Loads an audio file and returns the signal and sample rate. Also displays an audio player to play original sound

    Parameters
    ----------
    audio_path: string
        path to audio file to load

    Output
    ----------
    chord,sr: tuple
        contains the signal as index 0 and the sample rate as index 1
    """
    chord,sr = librosa.load(audio_path, sr=None)
    return (chord, sr)

def magnitude_spectrogram(signal, sr):
    """
    Displays a chromogram for a given signal

    Parameters
    ----------
    signal: 1D Numpy Array
        contains the original audio signal
    sr: int
        sample rate of the audio signal
    Output
    ----------
    FFT: 1D Numpy Array
        contains the magnitude spectrogram of the original signal
    """
    FFT = np.abs(np.fft.fft(signal))
    #fft_chroma_test = np.abs(fft_chroma_test)
    Nf = np.shape(FFT)[0]
    FFT = FFT[0:int(Nf/2)+1]
    return FFT

def hps_piano(mag_spec):
    """
    Calculates the harmonic product spectrum for a piano

    Parameters
    ----------
    mag_spec: 1D Numpy Array
        contains the magnitude spectrogram of the signal
    Output
    ----------
    hps: 1D Numpy Array
        contains the harmonic product spectrum
    """
    # decimates the audio 3 times
    dec1 = sp.signal.decimate(mag_spec, 1)
    dec2 = sp.signal.decimate(mag_spec, 2)
    dec3 = sp.signal.decimate(mag_spec, 4)

    # multiplies the decimated audio together to reduce overtones
    hps = np.zeros(len(dec3))
    for i in range(len(dec3)):
        product = dec1[i] * dec2[i] * dec3[i]
        hps[i] = product
    return hps

def hps_guitar(mag_spec):
    """
    Calculates the harmonic product spectrum for a guitar

    Parameters
    ----------
    mag_spec: 1D Numpy Array
        contains the magnitude spectrogram of the signal
    Output
    ----------
    hps: 1D Numpy Array
        contains the harmonic product spectrum
    """
    # decimates the audio signal 5 times to reduce overtones
    dec1 = sp.signal.decimate(mag_spec, 1)
    dec2 = sp.signal.decimate(mag_spec, 2)
    dec3 = sp.signal.decimate(mag_spec, 4)
    dec4 = sp.signal.decimate(mag_spec, 8)
    dec5 = sp.signal.decimate(mag_spec, 16)

    # multiplies the decimated audio together to reduce overtones
    hps = np.zeros(len(dec5))
    for i in range(len(dec5)):
        product = dec1[i] * dec2[i] * dec3[i] * dec4[i] * dec5[i]
        hps[i] = product
    return hps

def chromagram(hps,type_sig,sr):
    """
    Generates a 12-bin chromagram from a harmonic product spectrum

    Parameters
    ----------
    hps: 1D Numpy Array
    bins: # of chromagram bins desired

    Output
    ----------
    chroma_vec: 1D Numpy Array with shape=(1,12)
    """

    # initialize output vector
    chroma_vec = np.zeros((12))

    if(type_sig == 'guitar'):
        num_dec = 2**4
    else:
        num_dec = 2**2

    #hps_max = np.amax(hps)
    #hps = hps/hps_max

    win_len = len(hps)
    freq_arr = np.arange(win_len) * ((sr/num_dec)/win_len)

    # do log2(freq) - floor(log2(freq)) for each element in hps
    # results in a # in range (0,1)
    for i in range(len(freq_arr)):
        if hps[i] <= 0:
            continue
        else:
            cbin = find_chroma_bin(freq_arr[i])
            chroma_vec[cbin] = chroma_vec[cbin] + hps[i]

    #scale chroma_vec between 0 and 1
    cmax = np.amax(chroma_vec)
    chroma_vec = chroma_vec / cmax

    # plot 12-bin chromagram
    #plt.plot(np.linspace(0,11,12), chroma_vec, 'bo')
    #plt.xlabel("Note Number")
    #plt.show()

    return chroma_vec

def find_chroma_bin(freq):
    bin = 0
    i = np.log2(freq) - np.floor(np.log2(freq))
    if i < 1.0/12:
        bin = 0
    elif i>=1.0/12 and i<2.0/12:
        bin = 1
    elif i>=2.0/12 and i<3.0/12:
        bin = 2
    elif i>=3.0/12 and i<4.0/12:
        bin = 3
    elif i>=4.0/12 and i<5.0/12:
        bin = 4
    elif i>=5.0/12 and i<6.0/12:
        bin = 5
    elif i>=6.0/12 and i<7.0/12:
        bin = 6
    elif i>=7.0/12 and i<8.0/12:
        bin = 7
    elif i>=8.0/12 and i<9.0/12:
        bin = 8
    elif i>=9.0/12 and i<10.0/12:
        bin = 9
    elif i>=10.0/12 and i<11.0/12:
        bin = 10
    elif i>=11.0/12:
        bin = 11
    return bin

def knn(data_X, data_Y, query_X, dist_measure, k):
    query_rows = query_X.shape[0]
    data_rows = data_X.shape[0]
    cols = query_X.shape[1]
    res = np.zeros(query_rows)

    #iterate through each test example
    for i in range(query_rows):
        dist_arr = [None] * data_rows
        #iterate through the data examples and add the distance between each point and the test points to the distance array
        #as an array of tuples (index,distance)
        for j in range(data_rows):
            if dist_measure == 'euclidean':
                dist_arr[j] = (j,sp.spatial.distance.euclidean(query_X[i,:],data_X[j,:],2))
            elif dist_measure == 'cosine':
                dist_arr[j] = (j,sp.spatial.distance.cosine(query_X[i,:],data_X[j,:]))

        #Sort the array in ascending order by distance
        sorted_arr = sorted(dist_arr, key=lambda x:x[1])
        #Only need the first 'k' elements of the array
        k_neighbors = sorted_arr[0:k]
        classes = np.zeros(k)
        #create an array of class types corresponding to the classes of the elements in k_neighbors
        for l in range(k):
            index = k_neighbors[l][0]
            dclass = data_Y[index]
            classes[l] = dclass
        #Find the most common class type in classes[] by taking the mode
        closest = int(sp.stats.mode(classes)[0][0])
        res[i] = int(closest)
    return res

def create_key_time_vector(chord_progression):
    chords_vec = np.zeros(shape=(len(chord_progression),12))
    for i in range(len(chord_progression)):
        chords_vec[i,:] = chord_vectors[chord_progression[i]]
    key_time_vector = np.sum(chords_vec,axis=0)
    return key_time_vector

def ks(kt_vector):
    r_vector = np.zeros(24)
    res = np.array([None] * 3)
    for i in range(ks_key_set.shape[0]):
        r,p = sp.stats.pearsonr(ks_key_set[i,:],kt_vector)
        r_vector[i] = r
    key_index = np.argsort(-r_vector)[0:3]
    for i in range(len(key_index)):
        res[i] = key_dictionary[str(key_index[i])]
    return res

def calcRootDists(key, chord_progression):
    '''
    Inputs:
    key = String containing key
    chord_progression = Array of strings representing chord names

    Outputs:
    distVec = Array of computed distances between the roots of chords and the key center
    '''
    key_index = key_index_dictionary[key]
    distVec = np.zeros(len(chord_progression),dtype='int')
    for i in range(len(chord_progression)):
        root_index = root_index_dictionary[chord_progression[i][:2]]
        distVec[i] = np.mod(root_index-key_index,12, casting='same_kind')#(root_index - key_index) % 12
    return distVec

def generateRomanNumerals(distVec, key, chord_progression):
    dist_string_vec = np.array([None] * len(distVec))
    dist_quality_vec = np.array([None] * len(distVec))
    roman_numeral_vec = np.array([None] * len(distVec))
    for i in range(len(distVec)):
        if distVec[i] < 10:
            dist_string_vec[i] = '0' + str(distVec[i])
        else:
            dist_string_vec[i] = str(distVec[i])
        dist_quality_vec[i] = dist_string_vec[i] + str(chord_progression[i][2] + str(chord_progression[i][3]))

    for j in range(len(dist_quality_vec)):
        if 'Major' in key:
            roman_numeral_vec[j] = major_roman_dictionary[dist_quality_vec[j]]
        elif 'minor' in key:
            roman_numeral_vec[j] = minor_roman_dictionary[dist_quality_vec[j]]
    return roman_numeral_vec

def scoreRomanNums(roman_numeral_vec):
    score = 0
    for i in range(len(roman_numeral_vec)):
        if 'I' in roman_numeral_vec[i] or 'i' in roman_numeral_vec[i]:
            score = score + 5
        elif 'V' in roman_numeral_vec[i] or 'V7' in roman_numeral_vec[i] or 'viio' in roman_numeral_vec[i]:
            score = score + 3
        elif 'IV' in roman_numeral_vec[i] or 'iv' in roman_numeral_vec[i]:
            score = score + 1
        elif 'ii' in roman_numeral_vec[i] or 'II' in roman_numeral_vec[i]:
            score = score + 1

        if i == len(roman_numeral_vec) - 2:
            if 'V' in roman_numeral_vec[i] and 'I' in roman_numeral_vec[i+1]:
                score = score + 8
            if 'V' in roman_numeral_vec[i] and 'i' in roman_numeral_vec[i+1]:
                score = score + 8
        return score



def generate(path_name):
    path4 = './Audio/' + path_name
    sig_type = 'piano'
    chord, sr = load_audio(path4)


    times = librosa.onset.onset_detect(y=chord, sr=sr, units='samples')
    #print(times)
    energies = np.zeros(len(times))
    for i in range(len(times)):
        energies[i] = np.abs(chord[times[i]])

    #print(energies)
    #times = times[energies>=(.1*np.mean(energies))]
    #print(times)
    chord_progression = np.zeros(shape=(len(times),12))
    counter = 0

    # for each onset (chord) detected
    white_noise = []
    for i in times:
        # calculate magnitude spectrogram with values up to nyquist frequency and plot
        window_size = 6100
        spec = magnitude_spectrogram(chord[i:i+window_size], sr)
        sf = librosa.feature.spectral_flatness(y=chord[i:i+window_size], n_fft=window_size+1, hop_length=window_size+1)
        if(sf <= 10**-3):
            white_noise.append(counter)

            # use HPS algorithm to reduce overtones
            if(sig_type == 'guitar'):
                hps = hps_guitar(spec)
            else:
                hps = hps_piano(spec)

            # plot HPS output
            v = chromagram(hps,sig_type,sr)
            chord_progression[counter,:] = v
            # output information about chord analysis
            Nf = np.shape(hps)[0]
            indices = np.argsort(hps)
            if(sig_type == 'guitar'):
                indices = indices*((sr/16)/Nf)
            else:
                indices = indices*((sr/4)/Nf)
            length = np.shape(indices)[0]
            counter = counter + 1

    chord_progression = chord_progression[white_noise,:]
    labels = knn(training_chord_set, training_chord_labels,chord_progression, 'cosine',1)
    chord_names = [None] * len(labels)
    for i in range(len(labels)):
        chord_names[i] = chord_dictionary[str(int(labels[i]))]
    chord_names = np.array(chord_names)
    #print(chord_names)
    key_time = create_key_time_vector(chord_names)
    keys = ks(key_time)
    #print('Key: ',key,', Progression: ', chord_names)
    dists = np.zeros(shape=(len(keys),len(chord_names)),dtype='int')
    roman = np.empty(shape=(len(keys),len(chord_names)),dtype='object')
    scores = np.zeros(3)
    for i in range(len(keys)):
        dists[i,:] = calcRootDists(keys[i],chord_names)
        roman[i,:] = generateRomanNumerals(dists[i,:],keys[i],chord_names)
        scores[i] = scoreRomanNums(roman[i])

    roman_progression = roman[np.argmax(scores),:]
    key = keys[np.argmax(scores)]
    #return[key,roman_progression]
    return [key, roman_progression]
    #dists = calcRootDists(key,chord_names)
    #generateRomanNumerals(dists,key,chord_names)

if __name__ == "__main__":
    generate(sys.argv[1])
