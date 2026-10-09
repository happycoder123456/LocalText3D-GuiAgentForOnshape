; Inno Setup script — Onshape Agent
; Builds installer\Output\Onshape-Agent-Setup.exe
; Payload: source tree (agent\ + requirements) + install step that creates a
; venv, installs deps + Chromium, and makes Desktop/Start Menu shortcuts.

#define MyAppName "Onshape Agent"
#define MyAppVersion "1.0.0"
#define MyAppPublisher "LocalText3D"
#define MyAppExeName "Onshape Agent"

[Setup]
AppId={{8E4B6D42-9F3C-4A2B-9D7A-1C3E5F7A9B21}}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={localappdata}\Programs\OnshapeAgent
DefaultGroupName=Onshape Agent
DisableProgramGroupPage=yes
LicenseFile=..\LICENSE
OutputDir=Output
OutputBaseFilename=Onshape-Agent-Setup
SetupIconFile=..\agent_icon.ico
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
UninstallDisplayIcon={app}\agent_icon.ico
ArchitecturesInstallIn64BitMode=x64compatible

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
Source: "..\agent\*.py"; DestDir: "{app}\agent"; Flags: ignoreversion recursesubdirs
Source: "..\agent_icon.ico"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\requirements.txt"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\README.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\LICENSE"; DestDir: "{app}"; Flags: ignoreversion
Source: "payload\install_app.bat"; DestDir: "{app}\payload"; Flags: ignoreversion

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\.venv\Scripts\pythonw.exe"; Parameters: "-m agent gui"; WorkingDir: "{app}"; IconFilename: "{app}\agent_icon.ico"; Comment: "Local GUI agent that 3D-models in Onshape"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\.venv\Scripts\pythonw.exe"; Parameters: "-m agent gui"; WorkingDir: "{app}"; IconFilename: "{app}\agent_icon.ico"; Tasks: desktopicon; Comment: "Local GUI agent that 3D-models in Onshape"

[Run]
Filename: "{sys}\cmd.exe"; Parameters: "/c ""{app}\payload\install_app.bat"""; WorkingDir: "{app}"; Flags: runhidden waituntilterminated skipifdoesntexist; StatusMsg: "Preparing Python environment (first install takes a few minutes)…"
Filename: "{app}\.venv\Scripts\pythonw.exe"; Parameters: "-m agent gui"; WorkingDir: "{app}"; Description: "{cm:LaunchProgram,Onshape Agent}"; Flags: postinstall nowait skipifdoesntexist

[UninstallDelete]
Type: filesandordirs; Name: "{app}\.venv"
Type: filesandordirs; Name: "{app}\payload"


