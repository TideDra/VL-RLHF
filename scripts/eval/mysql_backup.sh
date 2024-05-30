cd ~/gozhang/code/VL-RLHF/scripts/eval/
mysqldump -uroot -pzgr2002411 -A -E -R --triggers > mm11.sql
curl -s -u 'tidedra:fTDYa5pgZx2f4P7y' -T mm11.sql 'https://ogi.teracloud.jp/dav/mysql/'
rm mm11.sql
