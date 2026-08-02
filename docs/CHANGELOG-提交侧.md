# CSIG-2026 提交侧开发记录

原仓库 `triple-mu/CSIG-2026` 分支 `master`，17 个提交，其中 **13 个从未推送** ——
`model_dir/hypir_weights.pth`（2.46 GB）被提交进历史，超过 GitHub 100 MB 单文件硬限，
`.git` 因此膨胀到 6.8 GB。源码已全部归档到本仓库 `submit/` 与 `analysis/`，
历史仅存 changelog；`.git` 本体在回收站。

```
0c8af1d 2026-08-01 pack: 输出改为 MMDDHHMM_描述.zip，便于把线上分数对回版本
2ef05b8 2026-08-01 后端默认改为 cuda（config.yaml），并修 bind() 对不可用后端的静默降级
8799ec6 2026-08-01 提交前收尾：恢复两处 import 守卫，auto 模式拒绝静默降级到纯 PyTorch
b6f9414 2026-08-01 csig_ops: attn_d512 挡掉 batch>1；新增分块尺寸基准
f72a78f 2026-08-01 csig_ops: 修 gn_stats 的占用率与访存合并，triton 后端追平 cuda
6f5fa0b 2026-08-01 csig_ops: 恢复 triton 缺失时的回退（被格式化工具删掉了）
2707d20 2026-08-01 runner: temb 折进 conv1.bias，整条 timestep 通路移出前向
49b634f 2026-08-01 bench: 加 nsys/ncu 采样脚本，窗口收在预热后的稳态迭代
572e984 2026-08-01 ops: 抽出 csig_ops.py 做算子分发，后端与内存布局绑定
cd01fc2 2026-08-01 runner: 拆分评测热路径与整图路径，内存布局改为加载期自选
663b843 2026-08-01 docs: 记录评分公式逆向与全部实验结论，清理临时文件
9b9f253 2026-07-31 runner: 改为纯 PyTorch（移除 Triton），custom_op 源码入库
9b82ac1 2026-07-30 runner: 自检 triton_attn_d512，在算错的架构上回退 SDPA
4589625 2026-07-30 custom_op_cpp_ext: 记录 Triton attn_d512 在 Volta 上失效，补 README
2dca82d 2026-07-30 custom_op_cpp_ext: 用 CUDA C++ 扩展替换四个 Triton 融合算子
b768d95 2026-07-30 fix hint
e44abff 2026-07-30 import scope
711dea0 2026-07-30 my impl
efc5801 2026-07-30 baseline
```
