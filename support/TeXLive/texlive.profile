selected_scheme scheme-infraonly

TEXDIR ./

TEXMFCONFIG $TEXMFSYSCONFIG
TEXMFHOME $TEXMFLOCAL
TEXMFLOCAL ./texmf-local
TEXMFSYSCONFIG ./texmf-config
TEXMFSYSVAR ./texmf-var
TEXMFVAR $TEXMFSYSVAR

tlpdbopt_install_docfiles 0
tlpdbopt_install_srcfiles 0
tlpdbopt_autobackup 0
