from obrg.core import calculate
if __name__ == '__main__':
    input_file = '../Stanford3dDataset_v1.2_Aligned_Version/Area_1/conferenceRoom_1/conferenceRoom_1.txt'
    output_file = './'
    calculate(input_file, output_file, debug=True, save=False)
