# def assembleData(trainingImages, trainingpatches, inputSize, dataset):
# #     data = []
# #     append = data.append
# #     label = []
# #     append = label.append
# #     for i in range(trainingImages):
# #         if not i % 10:
# #             time_end = time.time()
# #             print('time cost', time_end - time_start, 's')
# #         imgIdx = np.random.randint(0, dataset.size())
# #         imgBGR = dataset.getBGR(imgIdx)
# #         imgObj = dataset.getObj(imgIdx)
# #         width = np.shape(imgBGR)[1]
# #         height = np.shape(imgBGR)[0]
# #         for j in range(trainingpatches):
# #             data_ij = np.zeros([inputSize, inputSize, 3])
# #             x = np.random.randint(inputSize/2, width - inputSize/2)
# #             y = np.random.randint(inputSize/2, height - inputSize/2)
# #             data_ij = imgBGR[int(y - inputSize/2): int(y + inputSize/2), int(x - inputSize/2): int(x + inputSize/2), :]
# #             data.append(data_ij)
# #             label.append(imgObj[y][x]/1000.0)
# #     data = np.array(data)
# #     label = np.array(label)
# #     return data, label
# #
# #
# # def assembleBatch(offset, size, permutation, data, label):
# #     batchData = data[permutation[offset]:permutation[offset + size], :, :, :]
# #     batchLabels = label[permutation[offset]:permutation[offset + size], :]
# #     return batchData, batchLabels