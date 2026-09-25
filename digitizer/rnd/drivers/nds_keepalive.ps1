# nds_keepalive.ps1 — сторож драйверов (20.09, правило заказчика «чтобы время не терялось»).
# Раз в 10 минут (задача nds_keepalive): для каждого драйвера — если в его логе нет маркера конца и задача
# планировщика не Running, запустить задачу заново. Драйверы идемпотентны (супервизор, .part.pkl, снимки),
# чужие живые шарды они подхватывают, а не дублируют. ⚠ UTF-8 С BOM; маркеры — ASCII.
$log = 'F:\nds\output\taskS\_keepalive.log'
function Say($m) { "{0}  {1}" -f (Get-Date -Format 'MM-dd HH:mm:ss'), $m | Out-File -FilePath $log -Encoding utf8 -Append }
$jobs = @(
  @{ task='nds_slot_ab';   log='F:\nds\output\taskS\_slot_ab.log';   done='=== SLOT DONE ===' },
  @{ task='nds_names_ab';   log='F:\nds\output\taskS\_names_ab.log';  done='=== NAMES DONE ===' },
  @{ task='nds_train_bg1'; log='F:\nds\output\taskS\_train_bg1.log'; done='=== TRAIN DONE ===' },
  @{ task='nds_bg1_ab';    log='F:\nds\output\taskS\_bg1_ab.log';    done='=== BG1 DONE ===' }
)
foreach ($j in $jobs) {
  $t = Get-ScheduledTask -TaskName $j.task -ErrorAction SilentlyContinue
  if (-not $t) { continue }
  $finished = (Test-Path $j.log) -and (Select-String -Path $j.log -Pattern $j.done -SimpleMatch -Quiet)
  if ($finished) { continue }
  if ($t.State -eq 'Running') { continue }
  Say "$($j.task): не Running и маркера конца нет — запускаю заново (state=$($t.State))"
  Start-ScheduledTask -TaskName $j.task
}
