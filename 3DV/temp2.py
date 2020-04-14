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

# def assembleData(imageCount, hypsPerImage, objInputSize, rgbInputSize, dataset, model, temperature):
#     data = []
#     label = []
#     gp = properties.GlobalProperties()
#     camMat = gp.getCamMat()
#     for i in range(imageCount):
#         imgIdx = np.random.randint(0, dataset.size())
#         imgBGR = dataset.getBGR(imgIdx)
#         info = dataset.getInfo(imgIdx)
#         sampling = cnn.stochasticSubSample(imgBGR, objInputSize, rgbInputSize)
#         # Through the trained network, get the estimated object coordinate
#         estObj = cnn.getCoordImg(imgBGR, sampling, rgbInputSize, model)
#         poseGT = Hypothesis.Hypothesis()
#         poseGT.Info(info)
#         for h in range(hypsPerImage):
#             driftLevel = np.random.randint(0, 3)
#             if not driftLevel:
#                 poseNoise = poseGT * getRandHyp(2, 2)
#             else:
#                 poseNoise = poseGT * getRandHyp(10, 100)
#             # Construct data and label
#             # input: reprojection error image
#             data.append(cnn.getDiffMap(TYPE.our2cv([poseNoise.getRotation(), poseNoise.getTranslation()]),
#                                        estObj, sampling, camMat))
#             label.append(-1 * temperature * max(poseGT.calcAngularDistance(poseNoise),
#                                                np.linalg.norm(poseGT.getTranslation() - poseNoise.getTranslation())))
#     data = np.array(data)
#     label = np.array(label)
#     return data, label
#
# def assembleBatch(offset, size, permutation, data, label):
#     batchData = data[permutation[offset]:permutation[offset + size], :, :, :]
#     batchLabels = label[permutation[offset]:permutation[offset + size], :]
#     return batchData, batchLabels

# patches = []
# transform = Transform_OBJ()
# modeImg = np.zeros([np.shape(sampling)[0], np.shape(sampling)[1], 3])
# width = np.shape(colorData)[1]
# height = np.shape(colorData)[0]
# for x in range(np.shape(modeImg)[1]):
#     for y in range(np.shape(modeImg)[0]):
#         origX = sampling[y][x][0]
#         origY = sampling[y][x][1]
#         if origX < patchsize/2 or origY < patchsize/2 or origX > width-patchsize/2 or origY > height-patchsize/2:
#             continue
#         minX = int(origX - patchsize/2)
#         minY = int(origY - patchsize/2)
#         maxX = int(origX + patchsize/2)
#         maxY = int(origY + patchsize/2)
#         patch = colorData[minY:maxY, minX:maxX, :]
#         patches.append(transform(patch))
