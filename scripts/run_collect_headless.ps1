# 采集脚本 - 自动设置 ENABLE_CAMERAS=1 防止 headless 下相机黑屏
# 用法: .\scripts\run_collect_headless.ps1 level1_pick_noise_universal_all_obj_fail_200success
# 或: .\scripts\run_collect_headless.ps1 level1_pick_noise_universal_all_obj_fail_200success --no-video
# 加速: .\scripts\run_collect_headless.ps1 level1_pick_noise_universal_all_obj_fail_200success_fast --no-video --fast-sim

$env:ENABLE_CAMERAS = "1"
$config = if ($args[0]) { $args[0] } else { "level1_pick_noise_universal_all_obj_fail_200success" }
$extraArgs = @("--config-name", $config, "--headless")
if ($args.Length -gt 1) { $extraArgs += $args[1..($args.Length-1)] }
python main.py @extraArgs
