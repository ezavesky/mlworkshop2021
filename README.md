# Machine Learning Workshop 2021

This is where it all begins...
- Check out the [main workshop event page](https://INFO_SITE/cdo/events/internal-events/90dea45e-1454-11ec-8dca-7d45a6b8dd2a) for more information.

## Workshop prerequisites

To access the environments you will need the upstart role "AIaaS - Data User"  and "Pinnacle User" - we will not be able to accommodate same-day requests.

### Data

- You can access the ADLSg2 datastore in a couple of ways...
    - [azcopy](https://wiki.SERVICE_SITE/x/BAaPW) 
      (or the alternate [storage explorer](https://wiki.SERVICE_SITE/x/oAePW))
      - [https://STORAGE/mlworkshop2021]
    - [databricks](https://FORWARD_SITE/cdo-databricks) 
      - [abfss://mlworkshop2021@STORAGE/mlflow_imdb/IMDB_Dataset.csv]
- Need a general primer for ADLSg2 acccess (one of the new KM/gold replacements)?
    - [https://INFO_SITE/sites/data/SitePages/Access-the-CDO-Data-Platform's-Gold-Data-in-ADLSGen2.aspx]
- The primary data for this workshop is avaialble [via the Azure portal](https://PORTAL/#blade/Microsoft_Azure_Storage/ContainerMenuBlade/overview/storageAccountId/%2Fsubscriptions%2F81b4ec93-f52f-4194-9ad9-57e636bcd0b6%2FresourceGroups%2Fblackbird-prod-storage-rg%2Fproviders%2FMicrosoft.Storage%2FstorageAccounts%2Fblackbirdproddatastore/path/mlworkshop2021/etag/%220x8D9766DE75EA338%22/defaultEncryptionScope/%24account-encryption-key/denyEncryptionScopeOverride//defaultId//publicAccessVal/None) after Active Directory authentication.

## Starter code

Starter code is in this repository.
