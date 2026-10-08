; Per-user installer (no admin rights) for the onedir folder PoseCraft.spec builds.
; Built by scripts/build_installer.py (which passes the version, the flavour and, with --sign,
; the signing tool); by hand:
;   iscc /DAppVersion=0.1.0 /DFlavor=dml installer\PoseCraft.iss

#ifndef AppVersion
  #define AppVersion "0.1.0"
#endif
; The onedir build's runtime flavour (POSECRAFT_FLAVOR): dml (DirectML, any DirectX 12 GPU),
; cpu or cuda. Only names the setup file; the build in dist\PoseCraft is what gets installed.
#ifndef Flavor
  #define Flavor "dml"
#endif
#define AppName "PoseCraft"
#define AppExe "PoseCraft.exe"

#if !FileExists(AddBackslash(SourcePath) + "..\dist\PoseCraft\" + AppExe)
  #error dist\PoseCraft\PoseCraft.exe is missing: run `pyinstaller PoseCraft.spec` before compiling the installer.
#endif

[Setup]
AppId={{3C6E2A1D-7B44-4F0E-9D61-5A8C2F17B0E4}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher=MELEJ
DefaultDirName={localappdata}\Programs\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=..\dist\installer
OutputBaseFilename=PoseCraft-{#AppVersion}-{#Flavor}-setup
; scripts/build_installer.py --sign passes /DSign and the "codesign" tool: Inno Setup then
; signs this installer and the uninstaller it writes.
#ifdef Sign
SignTool=codesign
SignedUninstaller=yes
#endif
SetupIconFile=..\assets\icon.ico
UninstallDisplayIcon={app}\{#AppExe}
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
; Closes a running PoseCraft before an upgrade replaces its files.
CloseApplications=yes

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; Flags: unchecked

[InstallDelete]
; An upgrade replaces the whole runtime folder, so files a newer build no longer ships
; (e.g. after a flavour change) don't linger next to the new ones.
Type: filesandordirs; Name: "{app}\_internal"

[Files]
Source: "..\dist\PoseCraft\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\{#AppName}"; Filename: "{app}\{#AppExe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExe}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExe}"; Description: "Launch PoseCraft"; Flags: nowait postinstall skipifsilent

[Code]
// Settings (%APPDATA%\PoseCraft) and logs (%LOCALAPPDATA%\PoseCraft) are kept unless the user
// agrees here. SuppressibleMsgBox: a /SUPPRESSMSGBOXES silent uninstall keeps them (IDNO).
procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
begin
  if CurUninstallStep = usPostUninstall then
    if SuppressibleMsgBox('Also delete your PoseCraft settings and logs?', mbConfirmation, MB_YESNO or MB_DEFBUTTON2, IDNO) = IDYES then
    begin
      DelTree(ExpandConstant('{userappdata}\PoseCraft'), True, True, True);
      DelTree(ExpandConstant('{localappdata}\PoseCraft'), True, True, True);
    end;
end;
