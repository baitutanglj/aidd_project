# aidd_project
## 切记所以软件的安装、conda虚拟环境都装在"/share"目录下

### shape-it编译安装
shape-it的github链接：[shape-it](git@github.com:rdkit/shape-it.git)
- 编译安装所需依赖：openbabel
```
mkdir build
cd build
cmake   .. -DCMAKE_INSTALL_PREFIX=/shared/Programs/openbabel.3.1.1 \
-DZLIB_LIBRARY=/shared/Programs/anaconda3/lib/libz.so.1
make
sudo make install
```
- 编译安装shape-it(可能会报错，自己根据报错修改cmake的参数)
```
cmake .. -DCMAKE_INSTALL_PREFIX=/shared/Programs/shape-it \
-DOPENBABEL3_INCLUDE_DIR=/shared/Programs/openbabel.3.1.1/include/openbabel3 \
-DOPENBABEL3_LIBRARIES=/shared/Programs/openbabel.3.1.1/lib/libopenbabel.so
make
sudo make install
```
### conda 虚拟环境 environment.yml

### run_slurm.py:批量提交slurm任务，一个节点提交一个slurm任务
- morgan_similarity.py: 计算2D Fingerprint相似性
```
python run_slurm.py -p morgan_similarity.py  -f ./configs/linjie/morgan_similarity/0.json -d ./data/morgan_similarity
python run_slurm.py -p morgan_similarity.py  -f /shared/home/linjie/projects/AIDD/data/configs/morgan_similarity/0.json -d /shared/home/linjie/projects/AIDD/data/morgan_similarity
```
- sub_match.py： 搜索子结构
```
python run_slurm.py -p sub_match.py -f ./configs/linjie/sub_match/0.json -d ./data/sub_match
python run_slurm.py -p sub_match.py  -f /shared/home/linjie/projects/AIDD/data/configs/sub_match/0.json -d /shared/home/linjie/projects/AIDD/data/morgan_similarity
```
- shape_it_similarity.py: 计算3D相似性
```
python run_slurm.py -p shape_it_similarity.py  -f ./configs/linjie/shape_it/0.json -d ./data/shape_it
python run_slurm.py -p shape_it_similarity.py  -f /shared/home/linjie/projects/AIDD/data/configs/shape_it_similarity/0.json -d /shared/home/linjie/projects/AIDD/data/shape_it_similarity
```
- vina_docking.py: vina对接
```
python run_slurm.py -p vina_docking.py -f ./configs/linjie/vina_docking/0.json -d ./data/vina_docking/data
python run_slurm.py -p vina_docking.py -f /shared/home/linjie/projects/AIDD/data/configs/vina_docking/0.json -d /shared/home/linjie/projects/AIDD/data/vina_docking
```





