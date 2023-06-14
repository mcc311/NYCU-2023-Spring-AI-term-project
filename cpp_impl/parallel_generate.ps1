$n = Read-Host "Enter the value of n:"

for ($i = 0; $i -lt $n; $i++) {
    $command = "generate.exe -total=10000 -sim_count=10000 id=$i"
    Start-Process -FilePath generate.exe -ArgumentList "-total=10000", "-sim_count=10000", "id=$i"
}