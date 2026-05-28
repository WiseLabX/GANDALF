import pandas as pd
import os
import re

# =============================
# 1. CSV dosyasını oku
# =============================
file_path = '../../Kronodroid/real_malware_1-hybrid.csv'  # veya benign dosyan
# file_path = '../../Kronodroid/real_benign_1-hybrid.csv'  # veya benign dosyan
df = pd.read_csv(file_path)
cols = df.columns.tolist()

print("Toplam sütun:", len(cols))

# ============================
# 2. Manifest permission WHITELIST
# ============================

permission_whitelist = set([
    "ACCEPT_HANDOVER","ACCESS_BACKGROUND_LOCATION","ACCESS_CHECKIN_PROPERTIES",
    "ACCESS_COARSE_LOCATION","ACCESS_FINE_LOCATION","ACCESS_LOCATION_EXTRA_COMMANDS",
    "ACCESS_MEDIA_LOCATION","ACCESS_NETWORK_STATE","ACCESS_NOTIFICATION_POLICY",
    "ACCESS_WIFI_STATE","ACCOUNT_MANAGER","ACTIVITY_RECOGNITION","ADD_VOICEMAIL",
    "ANSWER_PHONE_CALLS","BATTERY_STATS","BIND_ACCESSIBILITY_SERVICE",
    "BIND_APPWIDGET","BIND_AUTOFILL_SERVICE","BIND_CALL_REDIRECTION_SERVICE",
    "BIND_CARRIER_MESSAGING_CLIENT_SERVICE","BIND_CARRIER_MESSAGING_SERVICE",
    "BIND_CARRIER_SERVICES","BIND_CHOOSER_TARGET_SERVICE",
    "BIND_CONDITION_PROVIDER_SERVICE","BIND_CONTROLS","BIND_DEVICE_ADMIN",
    "BIND_DREAM_SERVICE","BIND_INCALL_SERVICE","BIND_INPUT_METHOD",
    "BIND_MIDI_DEVICE_SERVICE","BIND_NFC_SERVICE","BIND_NOTIFICATION_LISTENER_SERVICE",
    "BIND_PRINT_SERVICE","BIND_QUICK_ACCESS_WALLET_SERVICE","BIND_QUICK_SETTINGS_TILE",
    "BIND_REMOTEVIEWS","BIND_SCREENING_SERVICE","BIND_TELECOM_CONNECTION_SERVICE",
    "BIND_TEXT_SERVICE","BIND_TV_INPUT","BIND_VISUAL_VOICEMAIL_SERVICE",
    "BIND_VOICE_INTERACTION","BIND_VPN_SERVICE","BIND_VR_LISTENER_SERVICE",
    "BIND_WALLPAPER","BLUETOOTH","BLUETOOTH_ADMIN","BLUETOOTH_PRIVILEGED",
    "BODY_SENSORS","BROADCAST_PACKAGE_REMOVED","BROADCAST_SMS","BROADCAST_STICKY",
    "BROADCAST_WAP_PUSH","CALL_COMPANION_APP","CALL_PHONE","CALL_PRIVILEGED",
    "CAMERA","CAPTURE_AUDIO_OUTPUT","CHANGE_COMPONENT_ENABLED_STATE",
    "CHANGE_CONFIGURATION","CHANGE_NETWORK_STATE","CHANGE_WIFI_MULTICAST_STATE",
    "CHANGE_WIFI_STATE","CLEAR_APP_CACHE","CONTROL_LOCATION_UPDATES",
    "DELETE_CACHE_FILES","DELETE_PACKAGES","DIAGNOSTIC","DISABLE_KEYGUARD",
    "DUMP","EXPAND_STATUS_BAR","FACTORY_TEST","FOREGROUND_SERVICE","GET_ACCOUNTS",
    "GET_ACCOUNTS_PRIVILEGED","GET_PACKAGE_SIZE","GET_TASKS","GLOBAL_SEARCH",
    "INSTALL_LOCATION_PROVIDER","INSTALL_PACKAGES","INSTALL_SHORTCUT",
    "INSTANT_APP_FOREGROUND_SERVICE","INTERACT_ACROSS_PROFILES","INTERNET",
    "KILL_BACKGROUND_PROCESSES","LOADER_USAGE_STATS","LOCATION_HARDWARE",
    "MANAGE_DOCUMENTS","MANAGE_EXTERNAL_STORAGE","MANAGE_OWN_CALLS",
    "MASTER_CLEAR","MEDIA_CONTENT_CONTROL","MODIFY_AUDIO_SETTINGS",
    "MODIFY_PHONE_STATE","MOUNT_FORMAT_FILESYSTEMS","MOUNT_UNMOUNT_FILESYSTEMS",
    "NFC","NFC_PREFERRED_PAYMENT_INFO","NFC_TRANSACTION_EVENT",
    "PACKAGE_USAGE_STATS","PERSISTENT_ACTIVITY","PROCESS_OUTGOING_CALLS",
    "QUERY_ALL_PACKAGES","READ_CALENDAR","READ_CALL_LOG","READ_CONTACTS",
    "READ_EXTERNAL_STORAGE","READ_INPUT_STATE","READ_LOGS","READ_PHONE_NUMBERS",
    "READ_PHONE_STATE","READ_PRECISE_PHONE_STATE","READ_SMS","READ_SYNC_SETTINGS",
    "READ_SYNC_STATS","READ_VOICEMAIL","REBOOT","RECEIVE_BOOT_COMPLETED",
    "RECEIVE_MMS","RECEIVE_SMS","RECEIVE_WAP_PUSH","RECORD_AUDIO",
    "REORDER_TASKS","REQUEST_COMPANION_RUN_IN_BACKGROUND",
    "REQUEST_COMPANION_USE_DATA_IN_BACKGROUND","REQUEST_DELETE_PACKAGES",
    "REQUEST_IGNORE_BATTERY_OPTIMIZATIONS","REQUEST_INSTALL_PACKAGES",
    "REQUEST_PASSWORD_COMPLEXITY","RESTART_PACKAGES","SEND_RESPOND_VIA_MESSAGE",
    "SEND_SMS","SET_ALARM","SET_ALWAYS_FINISH","SET_ANIMATION_SCALE",
    "SET_DEBUG_APP","SET_PREFERRED_APPLICATIONS","SET_PROCESS_LIMIT",
    "SET_TIME","SET_TIME_ZONE","SET_WALLPAPER","SET_WALLPAPER_HINTS",
    "SIGNAL_PERSISTENT_PROCESSES","SMS_FINANCIAL_TRANSACTIONS",
    "START_VIEW_PERMISSION_USAGE","STATUS_BAR","SYSTEM_ALERT_WINDOW",
    "TRANSMIT_IR","UNINSTALL_SHORTCUT","UPDATE_DEVICE_STATS","USE_BIOMETRIC",
    "USE_FINGERPRINT","USE_FULL_SCREEN_INTENT","USE_SIP","VIBRATE","WAKE_LOCK",
    "WRITE_APN_SETTINGS","WRITE_CALENDAR","WRITE_CALL_LOG","WRITE_CONTACTS",
    "WRITE_EXTERNAL_STORAGE","WRITE_GSERVICES","WRITE_SECURE_SETTINGS",
    "WRITE_SETTINGS","WRITE_SYNC_SETTINGS","WRITE_VOICEMAIL"
])

# ============================
# 3. Statik META sütunları
# ============================

STATIC_META = {
    "nr_permissions","normal","dangerous","signature","custom_yes","nr_custom",
    "total_perm","sha256","cfilesize","ufilesize","earliestmoddate",
    "highestmoddate","filesinsideapk","activities","nractivities",
    "nrintservices","nrintactivities","nrintreceivers","nrintreceiversactions",
    "totalintentfilters","nrservices","scanners","detection_ratio","malfamily"
}

# ============================
# 4. Ayırma işlemi
# ============================

static_features = []
dynamic_features = []
label_features = ["Package", "Malware", "MalFamily"]


for col in cols:

    if col in label_features:
        continue

    col_l = col.lower()

    # ----- STATİK -----
    if col in permission_whitelist:
        static_features.append(col)
        continue

    if col_l in STATIC_META:
        static_features.append(col)
        continue

    # ----- DİNAMİK -----
    if col.startswith("SYS_"):
        dynamic_features.append(col)
        continue

    if any(x in col_l for x in ["clock", "time", "timer"]):
        dynamic_features.append(col)
        continue

    if re.match(r"(exec|open|read|write|socket|send|recv|get|clone|fork|futex)", col_l):
        dynamic_features.append(col)
        continue

    # izin olmayan ALL CAPS ise dinamik
    if col.isupper() and col not in permission_whitelist:
        dynamic_features.append(col)
        continue

    # default → dinamik
    dynamic_features.append(col)

# Kesin ayrım
static_features = list(set(static_features) - set(dynamic_features))

print("Statik:", len(static_features))
print("Dinamik:", len(dynamic_features))

# ============================
# 5. Kaydetme
# ============================

os.makedirs("output_features", exist_ok=True)

df_static = df[static_features + label_features]
df_dynamic = df[dynamic_features + label_features]
df_hybrid = df[static_features + dynamic_features + label_features]

df_static.to_csv("output_features/real_malware_2-static_features.csv", index=False)
df_dynamic.to_csv("output_features/real_malware_2-dynamic_features.csv", index=False)
df_hybrid.to_csv("output_features/real_malware_2-hybrid_features.csv", index=False)

print("✔ CSV dosyaları başarıyla üretildi.")
