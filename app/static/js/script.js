document.addEventListener('DOMContentLoaded', function() {
    const fileInput = document.getElementById('fileInput');
    const uploadBox = document.getElementById('uploadBox');
    const diagnoseBtn = document.getElementById('diagnoseBtn');
    const resultSection = document.getElementById('resultSection');
    const originalImage = document.getElementById('originalImage');
    const heatmapImage = document.getElementById('heatmapImage');
    const diagnosisResult = document.getElementById('diagnosisResult');
    const confidence = document.getElementById('confidence');
    const uploadTime = document.getElementById('uploadTime');
    const historyBody = document.getElementById('historyBody');

    let selectedFile = null;

    // 监听文件选择
    fileInput.addEventListener('change', function(e) {
        if (e.target.files.length > 0) {
            selectedFile = e.target.files[0];
            diagnoseBtn.disabled = false;
            uploadBox.querySelector('span').textContent = `已选择文件: ${selectedFile.name}`;
        }
    });

    // 拖放功能
    uploadBox.addEventListener('dragover', function(e) {
        e.preventDefault();
        uploadBox.style.backgroundColor = '#f8f9fa';
        uploadBox.style.borderColor = '#2980b9';
    });

    uploadBox.addEventListener('dragleave', function() {
        uploadBox.style.backgroundColor = '';
        uploadBox.style.borderColor = '#3498db';
    });

    uploadBox.addEventListener('drop', function(e) {
        e.preventDefault();
        uploadBox.style.backgroundColor = '';
        uploadBox.style.borderColor = '#3498db';

        if (e.dataTransfer.files.length > 0) {
            const file = e.dataTransfer.files[0];
            if (file.type === 'image/png' || file.name.endsWith('.png')) {
                selectedFile = file;
                diagnoseBtn.disabled = false;
                uploadBox.querySelector('span').textContent = `已选择文件: ${selectedFile.name}`;
                fileInput.files = e.dataTransfer.files;
            } else {
                alert('仅支持PNG格式文件');
            }
        }
    });

    // 诊断按钮点击事件
    diagnoseBtn.addEventListener('click', function() {
        if (!selectedFile) return;

        const formData = new FormData();
        formData.append('file', selectedFile);

        // 显示加载状态
        diagnoseBtn.disabled = true;
        diagnoseBtn.textContent = '诊断中...';

        // 发送请求
        fetch('/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.code === 200) {
                // 显示结果
                resultSection.hidden = false;
                originalImage.src = data.data.image_url;
                heatmapImage.src = data.data.heatmap_url;
                diagnosisResult.textContent = data.data.result;
                diagnosisResult.className = `diagnosis-text ${data.data.result === '良性' ? 'benign' : 'malignant'}`;
                confidence.textContent = data.data.confidence;
                uploadTime.textContent = data.data.upload_time;

                // 刷新历史记录
                loadHistoryRecords();
            } else {
                alert(`诊断失败: ${data.msg}`);
            }
        })
        .catch(error => {
            console.error('诊断错误:', error);
            alert('诊断失败，请重试');
        })
        .finally(() => {
            diagnoseBtn.textContent = '开始诊断';
            diagnoseBtn.disabled = false;
        });
    });

    // 加载历史记录
    function loadHistoryRecords() {
        fetch('/records')
        .then(response => response.json())
        .then(data => {
            if (data.code === 200) {
                historyBody.innerHTML = '';
                data.data.forEach(record => {
                    const tr = document.createElement('tr');
                    tr.innerHTML = `
                        <td>${record.id}</td>
                        <td>${record.upload_time}</td>
                        <td><span class="${record.result === '良性' ? 'benign' : 'malignant'}">${record.result}</span></td>
                        <td>${record.confidence}</td>
                        <td><button class="view-btn" data-id="${record.id}">查看</button></td>
                        <td><button class="del-btn" data-id="${record.id}">删除</button></td>
                    `;
                    historyBody.appendChild(tr);
                });

                // 查看按钮事件
                document.querySelectorAll('.view-btn').forEach(btn => {
                    btn.addEventListener('click', function() {
                        const recordId = this.getAttribute('data-id');
                        fetch(`/records/${recordId}`)
                        .then(response => response.json())
                        .then(data => {
                            if (data.code === 200) {
                                resultSection.hidden = false;
                                originalImage.src = '/' + data.data.image_url;
                                heatmapImage.src = '/' + data.data.heatmap_url;
                                diagnosisResult.textContent = data.data.result;
                                diagnosisResult.className = `diagnosis-text ${data.data.result === '良性' ? 'benign' : 'malignant'}`;
                                confidence.textContent = data.data.confidence;
                                uploadTime.textContent = data.data.upload_time;
                            }
                        });
                    });
                });

                // 删除按钮事件,无需刷新页面
                document.querySelectorAll('.del-btn').forEach(btn => {
                    btn.addEventListener('click', function() {
                        const recordId = this.getAttribute('data-id');
                        // 2. 新增：删除确认弹窗（防止误操作）
                        if (!confirm(`确定要删除ID为 ${recordId} 的诊断记录吗？此操作不可恢复！`)) {
                            return; // 用户取消删除，直接退出
                        }
                        //删除对应的后端接口,method区分查找模块符合RESTful api规范
                        fetch(`/records/${recordId}`,{method:'DELETE',headers:{'Content-Type': 'application/json'}})
                        .then(response => response.json())
                        .then(data => {
                            //成功删除数据库后
                            if (data.code === 200) {
                                //清楚table中的该行记录
                                //从当前元素向上遍历DOM树找到最近的行
                                const currentRow = this.closest('tr');
                                if (currentRow) {
                                    currentRow.remove();
                                }
                                //弹窗删除成功
                                alert('记录删除成功！');
                            }

                        });
                    });
                });
            }
        });
    }

    // 初始化加载历史记录
    loadHistoryRecords();
});