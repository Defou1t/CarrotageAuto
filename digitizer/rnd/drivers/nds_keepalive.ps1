# nds_keepalive.ps1 — сторож драйверов (20.09, правило заказчика «чтобы время не терялось»).
# Раз в 10 минут (задача nds_keepalive): для каждого драйвера — если в его логе нет маркера конца и задача
# планировщика не Running, запустить задачу заново. Драйверы идемпотентны (супервизор, .part.pkl, снимки),
# чужие живые шарды они подхватывают, а не дублируют. ⚠ UTF-8 С BOM; маркеры — ASCII.
$log = 'F:\nds\output\taskS\_keepalive.log'
function Say($m) { "{0}  {1}" -f (Get-Date -Format 'MM-dd HH:mm:ss'), $m | Out-File -FilePath $log -Encoding utf8 -Append }
# ★ 26.09: отработавшие задачи сняты с планировщика (аудит проекта); в списке — только живые драйверы.
$jobs = @(
  @{ task='nds_seqbig';     log='F:\nds\output\taskS\_seqbig.log';    done='=== SEQBIG DONE ===' },
  @{ task='nds_sweep';      log='F:\nds\output\taskS\_sweep.log';     done='=== SWEEP DONE ===' },
  @{ task='nds_seqwide';    log='F:\nds\output\taskS\_seqwide.log';   done='=== SEQWIDE DONE ===' },
  @{ task='nds_seqhist';    log='F:\nds\output\taskS\_seqhist.log';   done='=== SEQHIST DONE ===' },
  @{ task='nds_knobab_hold'; log='F:\nds\output\taskS\_knobab_hold.log'; done='=== KNOBAB DONE ===' },
  @{ task='nds_gapfill';    log='F:\nds\output\taskS\_gapfill.log';   done='=== GAPFILL DONE ===' },
  @{ task='nds_holdfill';   log='F:\nds\output\taskS\_holdfill.log';  done='=== HOLDFILL DONE ===' },
  @{ task='nds_conf';       log='F:\nds\output\taskS\_conf.log';      done='=== CONF DONE ===' },
  @{ task='nds_veto';       log='F:\nds\output\taskS\_veto.log';      done='=== VETO DONE ===' },
  @{ task='nds_take';       log='F:\nds\output\taskS\_take.log';      done='=== TAKE DONE ===' },
  @{ task='nds_holdveto';   log='F:\nds\output\taskS\_holdveto.log';  done='=== HOLDVETO DONE ===' },
  @{ task='nds_knobab_embx'; log='F:\nds\output\taskS\_knobab_embx.log'; done='=== KNOBAB DONE ===' },
  @{ task='nds_upgrade';    log='F:\nds\output\taskS\_upgrade.log';   done='=== UPGRADE DONE ===' },
  @{ task='nds_rowdec_all'; log='F:\nds\output\taskS\_rowdec_all.log'; done='=== ROWDEC_ALL DONE ===' },
  @{ task='nds_rowdec_train'; log='F:\nds\output\taskS\_rowdec_train_f3.log'; done='=== TRAIN DONE ===' },
  @{ task='nds_rowdec_screen'; log='F:\nds\output\taskS\_rowdec_screen.log'; done='=== SCREEN DONE ===' }
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
