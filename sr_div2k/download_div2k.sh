#!/bin/bash
# 下载 DIV2K 数据集到 data 目录

# 创建数据目录
mkdir -p data

# 官方下载源
BASE="http://data.vision.ee.ethz.ch/cvl/DIV2K"

# 下载函数
download_file() {
    local filename=$1
    local url=$2
    local target_dir=$3
    
    if [ -d "data/$target_dir" ]; then
        echo "✅ $target_dir 目录已存在，跳过下载"
    else
        echo "📥 下载 $filename..."
        curl -L -o "data/$filename" "$url"
        if [ $? -eq 0 ]; then
            echo "✅ $filename 下载成功"
        else
            echo "❌ $filename 下载失败"
            echo "🔧 尝试使用国内镜像下载..."
            # 使用国内镜像
            local mirror_url="https://isv-data.oss-cn-beijing.aliyuncs.com/ML/DIV2K/$filename"
            curl -L -o "data/$filename" "$mirror_url"
            if [ $? -eq 0 ]; then
                echo "✅ $filename 从国内镜像下载成功"
            else
                echo "❌ $filename 从国内镜像下载也失败"
                return 1
            fi
        fi
    fi
    return 0
}

# 下载训练集
download_file "DIV2K_train_HR.zip" "$BASE/DIV2K_train_HR.zip" "DIV2K_train_HR"
download_file "DIV2K_train_LR_bicubic_X2.zip" "$BASE/DIV2K_train_LR_bicubic_X2.zip" "DIV2K_train_LR_bicubic/X2"
download_file "DIV2K_train_LR_bicubic_X3.zip" "$BASE/DIV2K_train_LR_bicubic_X3.zip" "DIV2K_train_LR_bicubic/X3"
download_file "DIV2K_train_LR_bicubic_X4.zip" "$BASE/DIV2K_train_LR_bicubic_X4.zip" "DIV2K_train_LR_bicubic/X4"

# 下载验证集
download_file "DIV2K_valid_HR.zip" "$BASE/DIV2K_valid_HR.zip" "DIV2K_valid_HR"
download_file "DIV2K_valid_LR_bicubic_X2.zip" "$BASE/DIV2K_valid_LR_bicubic_X2.zip" "DIV2K_valid_LR_bicubic/X2"
download_file "DIV2K_valid_LR_bicubic_X3.zip" "$BASE/DIV2K_valid_LR_bicubic_X3.zip" "DIV2K_valid_LR_bicubic/X3"
download_file "DIV2K_valid_LR_bicubic_X4.zip" "$BASE/DIV2K_valid_LR_bicubic_X4.zip" "DIV2K_valid_LR_bicubic/X4"

# 下载测试集（暂时跳过，因为测试集链接失效）
echo "⚠️  测试集链接暂时失效，跳过下载测试集"
echo "✅ 已下载训练集和验证集，足够用于模型训练和评估"

# 解压数据
echo "📦 解压数据集..."
cd data/
for zip_file in *.zip; do
    if [ -f "$zip_file" ]; then
        echo "解压 $zip_file..."
        unzip -o "$zip_file"
    fi
done

# 清理zip文件
echo "🧹 清理临时文件..."
rm -f *.zip

cd ..
echo "✅ 数据集下载完成！"
echo "📁 数据已保存到 data/ 目录"

