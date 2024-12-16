#!/bin/bash

log_group_name="/aws/lambda/mem_intensive_benchmark"
output_file="mem_intensive_log.csv"

# 기존 output.csv 파일 삭제
rm -f "$output_file"

# CSV 헤더 작성
echo "LogStreamName,Duration,BilledDuration,MemorySize,MaxMemoryUsed" >> "$output_file"

# 로그 스트림 이름을 최근 발생 순서대로 가져오기 (오름차순 정렬)
log_streams=$(aws logs describe-log-streams --log-group-name "$log_group_name" --order-by "LastEventTime" --descending --query 'logStreams[*].logStreamName' --output text)

# 각 로그 스트림에서 REPORT 로그 이벤트만 가져오기
for log_stream in $log_streams
do
    aws logs get-log-events --log-group-name "$log_group_name" --log-stream-name "$log_stream" --output text |
    grep "REPORT" |  # REPORT 로그만 가져옴
    awk '{
        # Duration 추출
        duration_index = index($0, "Duration: ");
        billed_duration_index = index($0, "Billed Duration: ");
        memory_size_index = index($0, "Memory Size: ");
        max_memory_used_index = index($0, "Max Memory Used: ");
        
        duration = substr($0, duration_index + 10, 8);
        billed_duration = substr($0, billed_duration_index + 17, 8);
        memory_size = substr($0, memory_size_index + 12, 8);
        max_memory_used = substr($0, max_memory_used_index + 17, 6);
        
        # 추출된 값을 CSV 형식으로 출력
        printf "%s,%s,%s,%s,%s\n", "'$log_stream'", duration, billed_duration, memory_size, max_memory_used;
    }' >> "$output_file"
done

echo "Filtered and formatted REPORT logs have been saved to $output_file"
