import os
import shutil
import util


def int2str(num, min_len = 6):
    string = str(num)
    while len(string) < min_len:
        string = '0' + string
    return string


def NameChange(filepath, num):
    file = sorted(os.listdir(filepath))
    n_RGB, n_Depth, n_pose = 0, 0, 0
    for i in file:
        if os.path.splitext(i)[1] == '.png':
            # Change name of RGB image
            if util.endsWith(os.path.splitext(i)[0], '.color'):
                oldname_RGB = filepath + os.path.splitext(i)[0] + '.png'
                newname_RGB = filepath + 'frame-' + int2str(n_RGB + int(1000 * num)) + '.color' + '.png'
                os.rename(oldname_RGB, newname_RGB)
                n_RGB += 1
            # Change name of depth image
            if util.endsWith(os.path.splitext(i)[0], '.depth'):
                oldname_Depth = filepath + os.path.splitext(i)[0] + '.png'
                newname_Depth = filepath + 'frame-' + int2str(n_Depth + int(1000 * num)) + '.depth' + '.png'
                os.rename(oldname_Depth, newname_Depth)
                n_Depth += 1
        else:
            oldname_pose = filepath + os.path.splitext(i)[0] + '.txt'
            newname_pose = filepath + 'frame-' + int2str(n_pose + int(1000 * num)) + '.pose' + '.txt'
            os.rename(oldname_pose, newname_pose)
            n_pose += 1


def Move_file(filepath):
    basepath = '/home/open/eth/2020spring/3DV/Project/Python/test/scene/'
    subpath = ['rgb_noseg/', 'depth_noseg/', 'poses/']
    file = sorted(os.listdir(filepath))
    for i in file:
        if os.path.splitext(i)[1] == '.png':
            # Move RGB file
            if util.endsWith(os.path.splitext(i)[0], '.color'):
                oldpath_RGB = filepath + os.path.splitext(i)[0] + '.png'
                newpath_RGB = basepath + subpath[0] + os.path.splitext(i)[0] + '.png'
                shutil.move(oldpath_RGB, newpath_RGB)
            # Move Depth file
            if util.endsWith(os.path.splitext(i)[0], '.depth'):
                oldpath_Depth = filepath + os.path.splitext(i)[0] + '.png'
                newpath_Depth = basepath + subpath[1] + os.path.splitext(i)[0] + '.png'
                shutil.move(oldpath_Depth, newpath_Depth)
        else:
            # Move pose file
            oldpath_pose = filepath + os.path.splitext(i)[0] + '.txt'
            newpath_pose = basepath + subpath[2] + os.path.splitext(i)[0] + '.txt'
            shutil.move(oldpath_pose, newpath_pose)


if __name__ == '__main__':
    basepath = '/home/open/eth/2020spring/3DV/Project/C++/DSAC/Data/chess/'
    subpath_train = ['seq-01/', 'seq-02/', 'seq-04/', 'seq-06/']
    subpath_test = ['seq-03/', 'seq-05/']
    for i in range(2):
        NameChange(basepath + subpath_test[i], i)
        Move_file(basepath + subpath_test[i])