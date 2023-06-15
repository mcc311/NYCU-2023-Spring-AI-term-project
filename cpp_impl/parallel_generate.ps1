$n = Read-Host "Enter the value of n:"
$sim_count = Read-Host "Enter the value of sim_count:"
$total = Read-Host "Enter the value of total:"

for ($i = 0; $i -lt $n; $i++) {
    Start-Process -FilePath generate.exe -ArgumentList "-sim_count=$sim_count", "id=$i", "total=$total"
    Start-Sleep -Seconds 2
}