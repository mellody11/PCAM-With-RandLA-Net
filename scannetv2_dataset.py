from helper_tool import DataProcessing as DP
from sklearn.neighbors import KDTree
from helper_tool import ConfigScanNet as cfg
from os.path import join
import numpy as np
import time, pickle, argparse, glob, os
from os.path import exists, join, isfile, isdir
from helper_ply import read_ply, write_ply
from torch.utils.data import DataLoader, Dataset, IterableDataset
import torch
from os import makedirs, listdir

# read the subsampled data and divide the data into training and validation
class Scannet(Dataset):
    def __init__(self, load_test=False):
        self.name = 'Scannet'
        self.path = '/data/dataset/scannet'

        self.train_path = join(self.path, 'training_points')
        self.test_path = join(self.path, 'test_points')

        self.label_to_names = {0: 'unclassified',
                               1: 'wall',
                               2: 'floor',
                               3: 'cabinet',
                               4: 'bed',
                               5: 'chair',
                               6: 'sofa',
                               7: 'table',
                               8: 'door',
                               9: 'window',
                               10: 'bookshelf',
                               11: 'picture',
                               12: 'counter',
                               14: 'desk',
                               16: 'curtain',
                               24: 'refridgerator',
                               28: 'shower curtain',
                               33: 'toilet',
                               34: 'sink',
                               36: 'bathtub',
                               39: 'otherfurniture'}
        self.num_classes = len(self.label_to_names)
        self.label_values = np.sort([k for k, v in self.label_to_names.items()])        # 进行升序排序,将列表转换为ndarray格式
        self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}             # {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 12}
        self.name_to_label = {v: k for k, v in self.label_to_names.items()}
        self.ignored_labels = np.array([0])                                              # 这个数据集上没有ignored标签

        cfg.ignored_label_inds = [self.label_to_idx[ign_label] for ign_label in self.ignored_labels]

        cfg.name = 'Scannet'
        cfg.class_weights = np.array([0])


        # Proportion of validation scenes
        self.validation_clouds = np.loadtxt(join(self.path, 'scannetv2_val.txt'), dtype=np.str)     # 有312个
        print(self.validation_clouds)
  
        # 1 to do validation, 2 to train on all data
        self.validation_split = 1
        self.training_split = 0
        self.all_splits = []

        # Load test set or train set?
        self.load_test = load_test

        self.prepare_pointcloud_ply()


        # Initiate containers
        self.val_proj = []
        self.val_labels = []
        self.test_proj = []
        self.test_labels = []

        self.possibility = {}
        self.min_possibility = {}
        
        self.input_trees = {'training': [], 'validation': [], 'test': []}
        self.input_colors = {'training': [], 'validation': [], 'test': []}
        self.input_labels = {'training': [], 'validation': []}
        self.input_names = {'training': [], 'validation': [], 'test': []}
        self.load_sub_sampled_clouds(cfg.sub_grid_size)

        print('Size of training : ', len(self.input_colors['training']))               
        print('Size of validation : ', len(self.input_colors['validation']))        

        self.num_classes = self.num_classes - len(self.ignored_labels)                  # 计算新的类别

    def prepare_pointcloud_ply(self):       # 处理原始的ply文件保存为新的ply文件，降采样，保存降采样之后的ply文件

        print('\nPreparing ply files')
        t0 = time.time()

        # Folder for the ply files
        paths = [join(self.path, 'scans'), join(self.path, 'scans_test')]
        new_paths = [self.train_path, self.test_path]
        mesh_paths = [join(self.path, 'training_meshes'), join(self.path, 'test_meshes')]


        # Mapping from annot to NYU labels ID
        label_files = join(self.path, 'scannetv2-labels.combined.tsv')
        with open(label_files, 'r') as f:
            lines = f.readlines()
            names1 = [line.split('\t')[1] for line in lines[1:]]
            IDs = [int(line.split('\t')[4]) for line in lines[1:]]
            annot_to_nyuID = {n: id for n, id in zip(names1, IDs)}

        for path, new_path, mesh_path in zip(paths, new_paths, mesh_paths):

            # Create folder
            if not exists(new_path):
                makedirs(new_path)
            if not exists(mesh_path):
                makedirs(mesh_path)

            # Get scene names
            scenes = np.sort([f for f in listdir(path)])
            N = len(scenes)

            for i, scene in enumerate(scenes):

                #############
                # Load meshes
                #############

                # Check if file already done
                if exists(join(new_path, scene + '.ply')):
                    continue
                t1 = time.time()

                # Read mesh
                vertex_data, faces = read_ply(join(path, scene, scene + '_vh_clean_2.ply'), triangular_mesh=True)
                vertices = np.vstack((vertex_data['x'], vertex_data['y'], vertex_data['z'])).T
                vertices_colors = np.vstack((vertex_data['red'], vertex_data['green'], vertex_data['blue'])).T

                vertices_labels = np.zeros(vertices.shape[0], dtype=np.int32)
                if new_path == self.train_path:

                    ## Load alignment matrix to realign points
                    #align_mat = None
                    #with open(join(path, scene, scene + '.txt'), 'r') as txtfile:
                    #    lines = txtfile.readlines()
                    #for line in lines:
                    #    line = line.split()
                    #    if line[0] == 'axisAlignment':
                    #        align_mat = np.array([float(x) for x in line[2:]]).reshape([4, 4]).astype(np.int32)
                    #R = align_mat[:3, :3]
                    #T = align_mat[:3, 3]
                    #vertices = vertices.dot(R.T) + T

                    # Get objects segmentations
                    with open(join(path, scene, scene + '_vh_clean_2.0.010000.segs.json'), 'r') as f:
                        segmentations = json.load(f)

                    segIndices = np.array(segmentations['segIndices'])

                    # Get objects classes
                    with open(join(path, scene, scene + '_vh_clean.aggregation.json'), 'r') as f:
                        aggregation = json.load(f)

                    # Loop on object to classify points
                    for segGroup in aggregation['segGroups']:
                        c_name = segGroup['label']
                        if c_name in names1:
                            nyuID = annot_to_nyuID[c_name]
                            if nyuID in self.label_values:
                                for segment in segGroup['segments']:
                                    vertices_labels[segIndices == segment] = nyuID

                    # Save mesh
                    write_ply(join(mesh_path, scene + '_mesh.ply'),
                              [vertices, vertices_colors, vertices_labels],
                              ['x', 'y', 'z', 'red', 'green', 'blue', 'class'],
                              triangular_faces=faces)

                else:
                    # Save mesh
                    write_ply(join(mesh_path, scene + '_mesh.ply'),
                              [vertices, vertices_colors],
                              ['x', 'y', 'z', 'red', 'green', 'blue'],
                              triangular_faces=faces)

                ###########################
                # Create finer point clouds 降采样
                ###########################

                # Rasterize mesh with 3d points (place more point than enough to subsample them afterwards)
                points, associated_vert_inds = rasterize_mesh(vertices, faces, 0.003)

                # Subsample points
                sub_points, sub_vert_inds = grid_subsampling(points, labels=associated_vert_inds, sampleDl=0.01)

                # Collect colors from associated vertex
                sub_colors = vertices_colors[sub_vert_inds.ravel(), :]

                if new_path == self.train_path:

                    # Collect labels from associated vertex
                    sub_labels = vertices_labels[sub_vert_inds.ravel()]

                    # Save points
                    write_ply(join(new_path, scene + '.ply'),
                              [sub_points, sub_colors, sub_labels, sub_vert_inds],
                              ['x', 'y', 'z', 'red', 'green', 'blue', 'class', 'vert_ind'])

                else:

                    # Save points
                    write_ply(join(new_path, scene + '.ply'),
                              [sub_points, sub_colors, sub_vert_inds],
                              ['x', 'y', 'z', 'red', 'green', 'blue', 'vert_ind'])

                #  Display
                print('{:s} {:.1f} sec  / {:.1f}%'.format(scene,
                                                          time.time() - t1,
                                                          100 * i / N))

        print('Done in {:.1f}s'.format(time.time() - t0))


    def load_sub_sampled_clouds(self, sub_grid_size):
        tree_path = join(self.path, 'input_{:.3f}_new'.format(sub_grid_size))
        if not exists(tree_path):
            makedirs(tree_path)

        # List of training files
        self.train_files = np.sort([join(self.train_path, f) for f in listdir(self.train_path) if f[-4:] == '.ply'])

        # Add test files
        self.test_files = np.sort([join(self.test_path, f) for f in listdir(self.test_path) if f[-4:] == '.ply'])
        files = np.hstack((self.train_files, self.test_files))      # 将所有文件集合起来了

        N = len(files)
        progress_n = 30
        fmt_str = '[{:<' + str(progress_n) + '}] {:5.1f}%'         # 进度条相关
        print('\nPreparing KDTree for all scenes, subsampled at {:.3f}'.format(sub_grid_size))

        for i, file_path in enumerate(files):

            # Restart timer
            t0 = time.time()

            # get cloud name and split
            cloud_name = file_path.split('/')[-1][:-4]                                  # file_path：/data/dataset/scannet/training_points/scene0000_00.ply
            cloud_folder = file_path.split('/')[-2]
            if 'train' in cloud_folder:
                if cloud_name in self.validation_clouds:
                    self.all_splits += [1]
                    cloud_split = 'validation'
                else:
                    self.all_splits += [0]
                    cloud_split = 'training'
            else:
                cloud_split = 'test'

            if (cloud_split != 'test' and self.load_test) or (cloud_split == 'test' and not self.load_test):
                continue

            # Name of the input files
            KDTree_file = join(tree_path, '{:s}_KDTree.pkl'.format(cloud_name))
            sub_ply_file = join(tree_path, '{:s}.ply'.format(cloud_name))

            # Check if inputs have already been computed        # 检查kdtree文件是否已经存在，若不存在则需要生成(通常来说都是存在的，先前已经执行过subcloud了)
            if isfile(KDTree_file):

                # read ply with data
                data = read_ply(sub_ply_file)
                sub_colors = np.vstack((data['red'], data['green'], data['blue'])).T
                sub_vert_inds = data['vert_ind']
                if cloud_split == 'test':
                    sub_labels = None
                else:
                    sub_labels = data['class']

                # Read pkl with search tree
                with open(KDTree_file, 'rb') as f:
                    search_tree = pickle.load(f)

            else:           # 没有这个文件说明还没有subcloud，要再这里subcloud一次

                # Read ply file
                data = read_ply(file_path)
                points = np.vstack((data['x'], data['y'], data['z'])).T
                colors = np.vstack((data['red'], data['green'], data['blue'])).T
                if cloud_split == 'test':
                    int_features = data['vert_ind']
                else:
                    int_features = np.vstack((data['vert_ind'], data['class'])).T

                # Subsample cloud
                sub_points, sub_colors, sub_int_features = DP.grid_sub_sampling(points,
                                                                      features=colors,
                                                                      labels=int_features,
                                                                      grid_size=sub_grid_size)

                # Rescale float color and squeeze label
                sub_colors = sub_colors / 255
                if cloud_split == 'test':
                    sub_vert_inds = np.squeeze(sub_int_features)
                    sub_labels = None
                else:
                    sub_vert_inds = sub_int_features[:, 0]
                    sub_labels = sub_int_features[:, 1]

                # Get chosen neighborhoods
                search_tree = KDTree(sub_points, leaf_size=50)  # 叶子节点的数量 更改leaf_size不影响查询结果，但是会影响查询速度和存储构建的时间

                # Save KDTree
                with open(KDTree_file, 'wb') as f:
                    pickle.dump(search_tree, f)


                # Save ply
                if cloud_split == 'test':
                    write_ply(sub_ply_file,
                              [sub_points, sub_colors, sub_vert_inds],
                              ['x', 'y', 'z', 'red', 'green', 'blue', 'vert_ind'])
                else:
                    write_ply(sub_ply_file,
                              [sub_points, sub_colors, sub_labels, sub_vert_inds],
                              ['x', 'y', 'z', 'red', 'green', 'blue', 'class', 'vert_ind'])

            # Fill data containers
            self.input_trees[cloud_split] += [search_tree]          # input_trees字典中包含两个列表，列表中包含一个个kdtree对象
            self.input_colors[cloud_split] += [sub_colors]
            self.input_names[cloud_split] += [cloud_name]
            if cloud_split in ['training', 'validation']:           # 这个数据集test是没有标签的吗
                self.input_labels[cloud_split] += [sub_labels]      # input_labels列表中的每个元素都是单个场景下的标签

            print('', end='\r')
            print(fmt_str.format('#' * ((i * progress_n) // N), 100 * i / N), end='', flush=True)       # 进度条，可以学一下

        # Get number of clouds
        self.num_training = len(self.input_trees['training'])
        self.num_validation = len(self.input_trees['validation'])
        self.num_test = len(self.input_trees['test'])
        print('\n')
        print('number of training sample:', self.num_training)          # 这里输出都是一样的
        print('number of validation sample:', self.num_validation)      # 这里输出都是一样的
        print('number of test sample:', self.num_test)

        # for i, file_path in enumerate(self.all_files):              # 这个是以前的
        #     t0 = time.time()
        #     cloud_name = file_path.split('/')[-1][:-4]
        #     if self.val_split in cloud_name:                # 云名字(字符串)中是否有指定的区域名字（子字符串）
        #         cloud_split = 'validation'
        #     else:
        #         cloud_split = 'training'

        #     # Name of the input files
        #     kd_tree_file = join(tree_path, '{:s}_KDTree.pkl'.format(cloud_name))            # 读的是采样后的数据
        #     sub_ply_file = join(tree_path, '{:s}.ply'.format(cloud_name))

        #     data = read_ply(sub_ply_file)                                                   # data['red'] 就这么读出来的是一个一维向量，存放了所有red的颜色深度
        #     sub_colors = np.vstack((data['red'], data['green'], data['blue'])).T            # 得到一个n*3的矩阵        
        #     sub_labels = data['class']

        #     # Read pkl with search tree
        #     with open(kd_tree_file, 'rb') as f:
        #         search_tree = pickle.load(f)

        #     self.input_trees[cloud_split] += [search_tree]              # 列表加列表 表示 列表的拼接，input_trees字典中保存了两个列表，每个列表中的元素都是kdtree对象
        #     self.input_colors[cloud_split] += [sub_colors]
        #     self.input_labels[cloud_split] += [sub_labels]
        #     self.input_names[cloud_split] += [cloud_name]

        #     size = sub_colors.shape[0] * 4 * 7
        #     print('{:s} {:.1f} MB loaded in {:.1f}s'.format(kd_tree_file.split('/')[-1], size * 1e-6, time.time() - t0))
        i_val = 0
        i_test = 0

        # Advanced display
        N = self.num_validation + self.num_test
        print('', end='\r')
        print(fmt_str.format('#' * progress_n, 100), flush=True)
        print('\nPreparing reprojection indices for validation and test')

        for i, file_path in enumerate(files):

            # get cloud name and split
            cloud_name = file_path.split('/')[-1][:-4]
            cloud_folder = file_path.split('/')[-2]

            # Validation projection and labels
            if (not self.load_test) and 'train' in cloud_folder and cloud_name in self.validation_clouds:
                proj_file = join(tree_path, '{:s}_proj.pkl'.format(cloud_name))
                if isfile(proj_file):
                    with open(proj_file, 'rb') as f:
                        proj_inds, labels = pickle.load(f)
                else:
                    # Get original mesh
                    mesh_path = file_path.split('/')
                    mesh_path[-2] = 'training_meshes'
                    mesh_path = '/'.join(mesh_path)
                    vertex_data, faces = read_ply(mesh_path[:-4] + '_mesh.ply', triangular_mesh=True)
                    vertices = np.vstack((vertex_data['x'], vertex_data['y'], vertex_data['z'])).T
                    labels = vertex_data['class']

                    # Compute projection inds
                    proj_inds = np.squeeze(self.input_trees['validation'][i_val].query(vertices, return_distance=False))
                    proj_inds = proj_inds.astype(np.int32)

                    # Save
                    with open(proj_file, 'wb') as f:
                        pickle.dump([proj_inds, labels], f)

                self.val_proj += [proj_inds]
                self.val_labels += [labels]
                i_val += 1

            # Test projection
            if self.load_test and 'test' in cloud_folder:
                proj_file = join(tree_path, '{:s}_proj.pkl'.format(cloud_name))
                if isfile(proj_file):
                    with open(proj_file, 'rb') as f:
                        proj_inds, labels = pickle.load(f)
                else:
                    # Get original mesh
                    mesh_path = file_path.split('/')
                    mesh_path[-2] = 'test_meshes'
                    mesh_path = '/'.join(mesh_path)
                    vertex_data, faces = read_ply(mesh_path[:-4] + '_mesh.ply', triangular_mesh=True)
                    vertices = np.vstack((vertex_data['x'], vertex_data['y'], vertex_data['z'])).T
                    labels = np.zeros(vertices.shape[0], dtype=np.int32)

                    # Compute projection inds
                    proj_inds = np.squeeze(self.input_trees['test'][i_test].query(vertices, return_distance=False))
                    proj_inds = proj_inds.astype(np.int32)

                    with open(proj_file, 'wb') as f:
                        pickle.dump([proj_inds, labels], f)

                self.test_proj += [proj_inds]
                self.test_labels += [labels]
                i_test += 1

            print('', end='\r')
            print(fmt_str.format('#' * (((i_val + i_test) * progress_n) // N), 100 * (i_val + i_test) / N),
                  end='',
                  flush=True)
        print('\n')

    def __getitem__(self, idx):
        pass

    def __len__(self):
        # Number of clouds 
        return self.size


class ScannetSampler(Dataset):

    def __init__(self, dataset, split='training'):
        self.dataset = dataset
        self.split = split
        self.possibility = {}
        self.min_possibility = {}

        if split == 'training':
            self.num_per_epoch = cfg.train_steps * cfg.batch_size       
        elif split in ['validation', 'test']:
            self.num_per_epoch = cfg.val_steps * cfg.val_batch_size

        self.possibility[split] = []
        self.min_possibility[split] = []
        for i, tree in enumerate(self.dataset.input_colors[split]):
            self.possibility[split] += [np.random.rand(tree.data.shape[0]) * 1e-3]              # 随机生成可能性 为每个场景的每一个点都生成可能性
            self.min_possibility[split] += [float(np.min(self.possibility[split][-1]))]         # 选出每个场景下最小可能性的那个点
        # 这里求概率是为了随机地选取场景中的中心点，选取中心点后通过kdtree找到这个中心点周围的K个点（KNN）
        # 更新中心点及邻近点的possibility并将这些点送进网络中，以实现点的不重复选择
        # possibility的更新方式是在随机初始值的基础上累加一个值，该值与该点到中心点的距离有关，且距离越大，该值越小（详见main_S3DIS第146行）。
        # 通过这样更新possibility的方式，使得抽过的点仅有很小的可能被抽中，从而实现类似穷举的目的。

    def __getitem__(self, item):
        selected_pc, selected_labels, selected_idx, cloud_ind, cloud_label, cloud_labels_all = self.spatially_regular_gen(item, self.split)
        return selected_pc, selected_labels, selected_idx, cloud_ind, cloud_label, cloud_labels_all

    def __len__(self):
        
        return self.num_per_epoch
        # return 2 * cfg.val_batch_size


    def spatially_regular_gen(self, item, split):

        # Choose a random cloud         # 选择可能性最小的那个点所属的场景
        cloud_idx = int(np.argmin(self.min_possibility[split]))     

        # choose the point with the minimum of possibility in the cloud as query point  选择该场景下的最小概率的点作为查询点 point_ind是点的序号
        point_ind = np.argmin(self.possibility[split][cloud_idx])

        # Get all points within the cloud from tree structure   从kdtree中得到这个场景中的所有点的xyz坐标
        points = np.array(self.dataset.input_trees[split][cloud_idx].data, copy=False)

        # Center point of input region  从所有点中选出概率最低的点（索引用上面求得的） center_point形状为(1,3)
        center_point = points[point_ind, :].reshape(1, -1)

        # Add noise to the center point
        noise = np.random.normal(scale=cfg.noise_init / 10, size=center_point.shape)
        pick_point = center_point + noise.astype(center_point.dtype)                    # 添加噪声

        # Check if the number of points in the selected cloud is less than the predefined num_points
        if len(points) < cfg.num_points:                # KNN取点       len一个矩阵，返回行数
            # Query all points within the cloud
            queried_idx = self.dataset.input_trees[split][cloud_idx].query(pick_point, k=len(points))[1][0]
        else:
            # Query the predefined number of points
            queried_idx = self.dataset.input_trees[split][cloud_idx].query(pick_point, k=cfg.num_points)[1][0]
            


        # Shuffle index
        queried_idx = DP.shuffle_idx(queried_idx)       # 将序号进行重新打乱分配以便随机采样


        # Update the possibility of the selected points
        dists = np.sum(np.square((points[queried_idx] - pick_point).astype(np.float32)), axis=1)    # 计算每个点离中心点的距离
        delta = np.square(1 - dists / np.max(dists))    # 这里注意先乘除后加减。 很巧妙地计算更新概率的大小（离中心点越远，要加的概率就越小，越容易在下一次选中心的时候选中）

        # 不同数据集不同的更新方式
        if split != 'training':
            self.possibility[split][cloud_idx][queried_idx] += delta    # 这里应该是更新概率，让下一选中心点时不重复
            self.min_possibility[split][cloud_idx] = float(np.min(self.possibility[split][cloud_idx]))  # 更新该场景的最小概率
        else:
            self.possibility[split][cloud_idx][queried_idx] += 0.01      # 如果是训练数据的话，整个邻域更新同一个概率，因为训练时是以球邻域为单位进行分类的
            self.min_possibility[split][cloud_idx] = float(np.min(self.possibility[split][cloud_idx]))      


        # Get corresponding points and colors based on the index    # 得到打乱后的点的信息
        queried_pc_xyz = points[queried_idx]            
        queried_pc_xyz = queried_pc_xyz - pick_point    # 减去中心点，去中心化
        queried_pc_colors = self.dataset.input_colors[split][cloud_idx][queried_idx]        # 用打乱的顺序取点的颜色信息
        if split == 'test':
            queried_pc_labels = self.dataset.input_labels[split][cloud_idx][queried_idx]        # 测试的时候用所有点的标签
        else:
            queried_pc_labels = self.dataset.input_labels[split][cloud_idx][queried_idx]
            queried_pc_labels = np.array([self.dataset.label_to_idx[l] for l in queried_pc_labels])
            cloud_labels_idx = np.unique(queried_pc_labels)
            cloud_labels_idx = cloud_labels_idx[cloud_labels_idx!=0].astype('int32')            # 去掉0
            cloud_labels = np.zeros((1, self.dataset.num_classes))
            cloud_labels[0][cloud_labels_idx-1] = 1

            # 不知道这两个量要不要 要的
            cloud_labels_all = np.ones((len(queried_pc_labels), self.dataset.num_classes)) 
            cloud_labels_all = cloud_labels_all * cloud_labels      # 子云下所有点的标签（但是都是一样的，都是子云的标签，只不过重复了n次）



        # up_sampled with replacement
        if len(points) < cfg.num_points:    # 如果不够40960个点，就使用数据增强到这么多个点
            queried_pc_xyz, queried_pc_colors, queried_idx, queried_pc_labels = \
                DP.data_aug(queried_pc_xyz, queried_pc_colors, queried_pc_labels, queried_idx, cfg.num_points) 

            cloud_labels_all = np.ones((len(queried_pc_labels), self.dataset.num_classes))      # 重新计算
            cloud_labels_all = cloud_labels_all * cloud_labels     

        queried_pc_xyz = torch.from_numpy(queried_pc_xyz).float()           # 转换回张量格式
        queried_pc_colors = torch.from_numpy(queried_pc_colors).float()
        queried_pc_labels = torch.from_numpy(queried_pc_labels).long()
        queried_idx = torch.from_numpy(queried_idx).float() # keep float here?
        cloud_idx = torch.from_numpy(np.array([cloud_idx], dtype=np.int32)).float()
        cloud_labels = torch.from_numpy(cloud_labels).long()
        cloud_labels_all = torch.from_numpy(cloud_labels_all).long()

        points = torch.cat( (queried_pc_xyz, queried_pc_colors), 1)
    
        return points, queried_pc_labels, queried_idx, cloud_idx, cloud_labels, cloud_labels_all      


    def tf_map(self, batch_xyz, batch_features, batch_label, batch_pc_idx, batch_cloud_idx, batch_cloud_label, batch_cloud_label_all):    # 进行下采样和KNN的索引记录，为后面网络做准备
        batch_features = np.concatenate([batch_xyz, batch_features], axis=-1)
        input_points = []
        input_neighbors = []
        input_pools = []
        input_up_samples = []

        for i in range(cfg.num_layers):     # 每一层的降采样在这里实现（从这里开始不可以再随意打乱矩阵的顺序了，因为knn search依靠的是矩阵的索引找到近邻点）
            neighbour_idx = DP.knn_search(batch_xyz, batch_xyz, cfg.k_n)      # KNN搜索每个点周围16个点，记录点的索引，维度是（6，40960，16）
            sub_points = batch_xyz[:, :batch_xyz.shape[1] // cfg.sub_sampling_ratio[i], :]      # 随机下采样 维度是（6，40690//4，3）
            pool_i = neighbour_idx[:, :batch_xyz.shape[1] // cfg.sub_sampling_ratio[i], :]      # 对索引也随机下采样 （6，40960//4，16）
            up_i = DP.knn_search(sub_points, batch_xyz, 1)                      # KNN搜索每个原点最近的下采样点 维度是（6，40960，1）
            input_points.append(batch_xyz)
            input_neighbors.append(neighbour_idx)
            input_pools.append(pool_i)
            input_up_samples.append(up_i)
            batch_xyz = sub_points

        input_list = input_points + input_neighbors + input_pools + input_up_samples
        input_list += [batch_features, batch_label, batch_pc_idx, batch_cloud_idx]
        input_list += [np.squeeze(batch_cloud_label, 1)]
        input_list += [batch_cloud_label_all]

        return input_list

    # 这个函数是每从dataloader拿一次数据执行一次
    def collate_fn(self,batch):

        selected_pc, selected_labels, selected_idx, cloud_ind, cloud_label, cloud_labels_all = [],[],[],[],[],[]
        for i in range(len(batch)):
            selected_pc.append(batch[i][0])
            selected_labels.append(batch[i][1])
            selected_idx.append(batch[i][2])
            cloud_ind.append(batch[i][3])
            cloud_label.append(batch[i][4])
            cloud_labels_all.append(batch[i][5])

        selected_pc = np.stack(selected_pc)                     # 在这里堆积成一个个batch
        selected_labels = np.stack(selected_labels)
        selected_idx = np.stack(selected_idx)
        cloud_ind = np.stack(cloud_ind)
        cloud_label = np.stack(cloud_label)
        cloud_labels_all = np.stack(cloud_labels_all)

        selected_xyz = selected_pc[:, :, 0:3]
        selected_features = selected_pc[:, :, 3:6]

        flat_inputs = self.tf_map(selected_xyz, selected_features, selected_labels, selected_idx, cloud_ind, cloud_label, cloud_labels_all) # 返回值是一个包含24个列表的列表

        num_layers = cfg.num_layers
        inputs = {}
        inputs['xyz'] = []
        for tmp in flat_inputs[:num_layers]:
            inputs['xyz'].append(torch.from_numpy(tmp).float())     # 添加了五个列表，每次随机采样前的坐标
        inputs['neigh_idx'] = []
        for tmp in flat_inputs[num_layers: 2 * num_layers]:
            inputs['neigh_idx'].append(torch.from_numpy(tmp).long())    # 添加了五个列表，输入点每次随机采样前的16个邻居的坐标（第一个列表没有进行下采样）
        inputs['sub_idx'] = []
        for tmp in flat_inputs[2 * num_layers:3 * num_layers]:
            inputs['sub_idx'].append(torch.from_numpy(tmp).long())      # 添加了五个列表，输入点的每次随机采样后的16个邻居的坐标
        inputs['interp_idx'] = []
        for tmp in flat_inputs[3 * num_layers:4 * num_layers]:
            inputs['interp_idx'].append(torch.from_numpy(tmp).long())   # 添加了五个列表，输入点每次随机采样后每个原点的最近的下采样点

        # inputs['features'] = torch.from_numpy(flat_inputs[4 * num_layers]).transpose(1,2).float()   # 转置了一下
        inputs['features'] = torch.from_numpy(flat_inputs[4 * num_layers]).float()  # 改了一下，为了适应后面linear的维度，不转置了
        inputs['labels'] = torch.from_numpy(flat_inputs[4 * num_layers + 1]).long()
        inputs['input_inds'] = torch.from_numpy(flat_inputs[4 * num_layers + 2]).long()
        inputs['cloud_inds'] = torch.from_numpy(flat_inputs[4 * num_layers + 3]).long()             # 维度为6，1
        inputs['cloud_label'] = torch.from_numpy(flat_inputs[4 * num_layers + 4]).float()            # 维度为6，1，13
        inputs['cloud_labels_all'] = torch.from_numpy(flat_inputs[4 * num_layers + 5]).float()

        return inputs


if __name__ == '__main__':
    dataset = Scannet()
    dataset_train = ScannetSampler(dataset, split='training')
    dataloader = DataLoader(dataset_train, batch_size=cfg.batch_size, shuffle=True, drop_last=True, collate_fn=dataset_train.collate_fn)
    # dataloader = DataLoader(dataset_train, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    for data in dataloader:

        features = data['features']
        labels = data['labels']
        idx = data['input_inds']
        cloud_idx = data['cloud_inds']
        print(features.shape)
        print(labels.shape)
        print(idx.shape)
        print(cloud_idx.shape)
        break

